import numpy as np
import torch
from einops import rearrange

from lavis.common.registry import registry
from lavis.models.sevila_models.sevila import SeViLAForVideoQA

from lavis.common.const import *

@registry.register_model("convila")
class ConViLAForVideoQA(SeViLAForVideoQA):
    """
    Extension of SeViLA using contrastive divergence loss for training the answerer and localizer in an end-to-end fashion
    """
    def __init__(
        self, 
        num_query_token=32, 
        num_keyframes=8,
        t5_model="google/flan-t5-xl", 
        task=CONVILA_TRAIN_E2E,
        num_negatives=3
    ):
        super(ConViLAForVideoQA, self).__init__(
            num_query_token=num_query_token,
            num_keyframes=num_keyframes,
            t5_model=t5_model,
            task=task
        )

        self.num_negatives = num_negatives

        # Unfreeze both answerer and localizer for end to end training
        if self.task == CONVILA_TRAIN_E2E:
            for param in self.Qformer_answerer.parameters():
                param.requires_grad = True
            self.query_tokens_answerer.requires_grad = True
            self.projection_answerer.requires_grad = True

            for param in self.Qformer_localizer.parameters():
                param.requires_grad = True
            self.query_tokens_localizer.requires_grad = True
            self.projection_localizer.requires_grad = True

    def contrastive_divergence_loss(self, loss : torch.Tensor):
        """
        Computes the contrastive divergence loss on a example-wise loss tensor of size (B * (num_negatives + 1))
        """
        # Reshape back to (B, num_negatives + 1)
        loss = rearrange(loss, "(b n) -> b n", n=(self.num_negatives + 1))
        negative_losses_sum, positive_losses = torch.sum(loss[:, 1:], dim=-1), loss[:, 1:]

        # Compute contrastive divergence
        loss_div = torch.div(positive_losses, negative_losses_sum)

        return loss_div


    def forward(self, samples,
        do_sample=False,
        num_beams=1, 
        max_new_tokens=30,
        min_length=1, 
        top_p=0.9,
        repetition_penalty=1.0, 
        length_penalty=1.0,
        num_return_sequences=1, 
        temperature=1
    ):
        video = samples["video"]

        # Extract visual tokens
        B, T, C, W, H = video.shape     
        video = rearrange(video, "b t c h w -> (b t) c h w")
        visual_tokens = self.visual_encoder(video) 
        visual_attention_mask = torch.ones(visual_tokens.size()[:-1], dtype=torch.long).to(video.device) # bt n c

        answerer_input_text, localizer_input_text, answer_text = samples['text_input'], samples['localizer_input'], samples['answer']

        if self.task == CONVILA_TRAIN_E2E:
            with torch.cuda.amp.autocast(dtype=self.dtype):
                # Generate localizer keyframe predictions
                localizer_outputs = self.generate_t5(
                    self.ln_vision_localizer(visual_tokens.detach().clone()), visual_attention_mask.detach().clone(),
                    localizer_input_text, None, # No output labels for localizer when generating
                    self.Qformer_localizer, self.query_tokens_localizer, self.projection_localizer,
                    video_shape=(B, T, C, H, W),
                    do_sample=do_sample, top_p=top_p, temperature=temperature,
                    max_new_tokens=max_new_tokens, min_length=min_length, 
                    repetition_penalty=repetition_penalty, length_penalty=length_penalty, 
                    num_return_sequences=num_return_sequences, num_beams=num_beams
                )

                localizer_logits = localizer_outputs.scores[0]
                positive_logits = localizer_logits[:, self.id_positive].reshape(B, T)

                visual_tokens = self.ln_vision_answerer(visual_tokens)
                _, N, D = visual_tokens.shape
                # Select top K frames
                topk_idx = torch.topk(positive_logits, self.num_keyframes, dim=-1).indices
                topk_idx = torch.tensor([sorted(idx.tolist()) for idx in topk_idx]).to(visual_tokens.device)
                visual_tokens = rearrange(visual_tokens, "(b t) n d -> b t n d", b=B, t=T)

                # Create mask from predicted logits. TODO: Try different mechanisms for this part
                # positive_logits = torch.nn.Sigmoid()(torch.exp(positive_logits)) # Extra step to inflate the weight to 1
                predicted_mask = positive_logits.reshape(B, T, 1, 1).repeat(1, 1, N, D)
                masked_tokens = torch.mul(predicted_mask, visual_tokens) # Apply mask multiplication to connect frames to localizer
                topk_visual_tokens = torch.stack([masked_tokens[i, row, :, :] for i, row in enumerate(topk_idx)]) # Index to get only top k frames

                # Select num_negative frames from the remaining frames
                remaining_idx = torch.tensor([
                    [i for i in range(T) if i not in row.tolist()] for row in topk_idx
                ]).to(visual_tokens.device)
                # Sample num_keyframes * num_negatives
                random_idx_order = torch.stack([
                    torch.randperm(remaining_idx.shape[1]) for _ in range(B)
                ])[:, :(self.num_keyframes * self.num_negatives)]
                negative_idx = torch.sort(random_idx_order, dim=-1).values
                negative_visual_tokens = torch.stack([visual_tokens[i, row, :, :] for i, row in enumerate(negative_idx)]) #(B, (num_negatives + 1) * num_keyframes, N, D)
                # Stack predicted and negative together
                sampled_visual_tokens = torch.cat([topk_visual_tokens, negative_visual_tokens], dim=1)
                sampled_visual_tokens = sampled_visual_tokens.reshape(
                    B, 1 + self.num_negatives, self.num_keyframes, N, D # (B, num_negatives + 1, num_keyframes, N, D)
                ).reshape((1 + self.num_negatives) * B, self.num_keyframes, N, D) # (B * (num_negatives + 1), num_keyframes, N, D)
                sampled_attention_mask = torch.ones(sampled_visual_tokens.shape[:-1], dtype=torch.long).to(video.device)

                # Repeat answer and input prompt to (B * (num_negative + 1))
                answer_text = np.repeat(answer_text, self.num_negatives + 1).tolist()
                answerer_input_text = np.repeat(answerer_input_text, self.num_negatives + 1).tolist()

                outputs = self.forward_t5(
                    sampled_visual_tokens, sampled_attention_mask,
                    answerer_input_text, answer_text,
                    self.Qformer_answerer, self.query_tokens_answerer, self.projection_answerer,
                    video_shape=(B, T, C, H, W),
                    reduction="none" 
                )
                loss = self.contrastive_divergence_loss(outputs.loss)

            return {
                "loss" : loss
            }

        elif self.task == CONVILA_REFINE_LOCALIZER:
            pass

    @classmethod
    def from_config(cls, cfg):
        num_query_token = cfg.get("num_query_token", 32)
        num_keyframes = cfg.get("num_keyframes", 4)
        num_negatives = cfg.get("num_negatives", 3)
        task = cfg.get("task")

        model = cls(
            num_query_token=num_query_token,
            num_keyframes=num_keyframes,
            num_negatives=num_negatives,
            task=task
        )
        model.load_checkpoint_from_config(cfg)

        return model
