import numpy as np
import torch
from torch import nn
from torch.distributions.gumbel import Gumbel
from einops import rearrange

from lavis.common.registry import registry
from lavis.models.sevila_models.sevila import SeViLAForVideoQA

from lavis.common.const import *

class GumbelTopK(nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(GumbelTopK, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau

    def forward(self, scores):
        m = Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores).to(scores.device)
        onehot_approx = torch.zeros_like(scores).to(scores.device)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([np.finfo(np.float32).tiny]).to(scores.device))
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res

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

        if self.task == CONVILA_TRAIN_ALTERNATE:
            self.train_answerer = False

    def toggle_training_mode(self):
        """
        Call before each training epoch, including the first one to determine which subnetwork to train
        """
        assert self.task == CONVILA_TRAIN_ALTERNATE, "Only call when using alternate training mode!"
        self.train_answerer = not self.train_answerer
        if self.train_answerer:
            for param in self.Qformer_answerer.parameters():
                param.requires_grad = True
            self.query_tokens_answerer.requires_grad = True
            self.projection_answerer.requires_grad = True

            for param in self.Qformer_localizer.parameters():
                param.requires_grad = False
            self.query_tokens_localizer.requires_grad = False
            self.projection_localizer.requires_grad = False
        else:
            for param in self.Qformer_answerer.parameters():
                param.requires_grad = False
            self.query_tokens_answerer.requires_grad = False
            self.projection_answerer.requires_grad = False

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
        negative_losses_sum, positive_losses = torch.sum(loss[:, 1:], dim=-1), loss[:, 1]

        # Compute contrastive divergence
        loss_div = torch.div(positive_losses, negative_losses_sum)

        return loss_div.mean()

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

                # Discretize positive_logits in a differentiable manner
                gumbel_topk = GumbelTopK(self.num_keyframes, hard=True)
                positive_logits = gumbel_topk(positive_logits)

                # Create mask from predicted logits. TODO: Try different mechanisms for this part
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

        elif self.task == CONVILA_TRAIN_ALTERNATE:
            # Fine-tune answerer using CD loss
            if self.train_answerer:
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
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
                topk_visual_tokens = torch.stack([visual_tokens[i, row, :, :] for i, row in enumerate(topk_idx)])

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

                with torch.cuda.amp.autocast(dtype=self.dtype):        
                    outputs = self.forward_t5(
                        sampled_visual_tokens, sampled_attention_mask,
                        answerer_input_text, answer_text,
                        self.Qformer_answerer, self.query_tokens_answerer, self.projection_answerer,
                        video_shape=(B, T, C, H, W),
                        reduction="none" 
                    )
                    loss = self.contrastive_divergence_loss(outputs.loss)

            # Refine localizer
            else:
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
                    # Perform forward using answerer QFormer
                    answerer_outputs = self.forward_t5(
                        self.ln_vision_answerer(visual_tokens.detach().clone()), visual_attention_mask.detach().clone(),
                        answerer_input_text, answer_text,
                        self.Qformer_answerer, self.query_tokens_answerer, self.projection_answerer,
                        video_shape=(B, T, C, H, W)
                    )

                    # Get predictions for answer set
                    answer_logits = answerer_outputs.logits.detach()[:, 1, self.answer_ids]
                    answer_preds = torch.argmax(answer_logits, dim=-1).reshape(B, T)
                    # Get pseudo-labels based on predicted answers
                    answer_labels = torch.tensor([self.answer_map[a[-1]] for a in answer_text]).reshape(-1, 1).repeat(1, T).to(answer_preds.device)
                    pseudo_labels = rearrange(answer_labels == answer_preds, "b t -> (b t)")
                    pseudo_labels = [
                        self.pseudo_label_positive if x else self.pseudo_label_negative for x in pseudo_labels.tolist()
                    ] # (B * T)
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    localizer_outputs = self.forward_t5(
                        self.ln_vision_localizer(visual_tokens), visual_attention_mask,
                        localizer_input_text, pseudo_labels,
                        self.Qformer_localizer, self.query_tokens_localizer, self.projection_localizer,
                        video_shape=(B, T, C, H, W)
                    )
                    loss = localizer_outputs.loss

        elif self.task == CONVILA_TRAIN_MULTITASK:
            # VideoQA objective
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
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
            # Select top K frames
            topk_idx = torch.topk(positive_logits, self.num_keyframes, dim=-1).indices
            topk_idx = torch.tensor([sorted(idx.tolist()) for idx in topk_idx]).to(visual_tokens.device)
            visual_tokens = rearrange(visual_tokens, "(b t) n d -> b t n d", b=B, t=T)
            topk_visual_tokens = torch.stack([visual_tokens[i, row, :, :] for i, row in enumerate(topk_idx)])
            topk_attention_mask = torch.ones(topk_visual_tokens.shape[:-1], dtype=torch.long).to(video.device)
            
            with torch.cuda.amp.autocast(dtype=self.dtype):        
                outputs = self.forward_t5(
                    topk_visual_tokens, topk_attention_mask,
                    answerer_input_text, answer_text,
                    self.Qformer_answerer, self.query_tokens_answerer, self.projection_answerer,
                    video_shape=(B, T, C, H, W)
                )
                qa_loss = outputs.loss

            # Frame order prediction objective
            _, _, N, D = topk_visual_tokens.shape
            frame_pred_input = samples['frame_pred_input']

            # Generate random frame permutations
            frame_permute_idx = torch.stack([torch.randperm(self.num_keyframes) for _ in range(B)]).to(video.device)
            shuffled_topk_visual_tokens = torch.gather(
                topk_visual_tokens, dim=1, index=frame_permute_idx.reshape(B, -1, 1, 1).repeat(1, 1, N, D))

            # Generate answers for frame order prediction task
            frame_pred_answer_text = [[self.frame_seq_prefix[i][:-1] for i in idx] for idx in frame_permute_idx]
            frame_pred_answer_text = [", ".join(o) for o in frame_pred_answer_text] # Frame X, Frame Y, ...

            # Share answerer for frame prediction
            with torch.cuda.amp.autocast(dtype=self.dtype):        
                outputs = self.forward_t5(
                    shuffled_topk_visual_tokens, topk_attention_mask,
                    frame_pred_input, frame_pred_answer_text,
                    self.Qformer_answerer, self.query_tokens_answerer, self.projection_answerer,
                    video_shape=(B, T, C, H, W)
                )
                frame_pred_loss = outputs.loss
        
        return {
            "loss" : qa_loss + frame_pred_loss
        }

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
