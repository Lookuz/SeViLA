"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from typing import Union

from einops import rearrange
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, LayerNorm, disabled_train
from lavis.models.eva_vit import VisionTransformer, create_eva_vit_g
from lavis.models.blip2_models.modeling_t5 import T5ForConditionalGeneration, T5Config

from lavis.common.const import *

@registry.register_model("sevila")
class SeViLA(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__( self, img_size=224, drop_path_rate=0,
        use_grad_checkpoint=False, vit_precision="fp16", freeze_vit=True,
        num_query_token=32, t5_model="google/flan-t5-xl", prompt="",
        max_txt_len=32, frame_num=8, answer_num=5, apply_lemmatizer=False, task='qa'):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        
        self.task = task

        # Initialize vision transformer feature extractor
        self.visual_encoder, self.ln_vision_answerer, self.ln_vision_localizer = self.init_vision_encoder()
        for param in self.visual_encoder.parameters(): # Freeze ViT weights
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train

        # Initialize T5 language model
        self.t5_model, self.t5_tokenizer = self.init_t5_model(t5_model=t5_model)
        for param in self.t5_model.parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        # Initialize Qformer for QA and localization
        self.Qformer_answerer, self.projection_answerer, self.query_tokens_answerer = self.init_Qformer(self.t5_model, num_query_token, self.visual_encoder.num_features)
        self.Qformer_localizer, self.projection_localizer, self.query_tokens_localizer = self.init_Qformer(self.t5_model, num_query_token, self.visual_encoder.num_features)

        if 'freeze_qa' in task:
            for param in self.Qformer_answerer.parameters():
                param.requires_grad = False
            self.query_tokens_answerer.requires_grad = False
            self.projection_answerer.requires_grad = False

        if 'freeze_loc' in task:
            for param in self.Qformer_localizer.parameters():
                param.requires_grad = False
            self.query_tokens_localizer.requires_grad = False
            self.projection_localizer.requires_grad = False

        answer_id = [71, 272, 205, 309, 262] # A B C D E
        self.answer_ids = answer_id[:answer_num]
        self.pseudo_label_positive = SEVILA_PSEUDO_LABEL_POSITIVE
        self.pseudo_label_negative = SEVILA_PSEUDO_LABEL_NEGATIVE
        self.id_positive = SEVILA_ID_POSITIVE
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        
        self.max_length = SEVILA_MAX_TEXT_LENGTH
        self.num_keyframes = frame_num
        self.answer_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        self.frame_prefix = ['Frame: ']
        self.frame_seq_prefix = self.repeat_frame_prefix(self.num_keyframes)

        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def init_vision_encoder(
        self, 
        img_size=224, 
        drop_path_rate=0, 
        use_grad_checkpoint=False, 
        precision="fp16"
    ) -> Union[VisionTransformer, LayerNorm, LayerNorm]:
        visual_encoder = create_eva_vit_g(img_size, drop_path_rate, use_grad_checkpoint, precision)
    
        # Layer normalization for answerer and localizer
        ln_vision_qa, ln_vision_localizer = LayerNorm(visual_encoder.num_features), LayerNorm(visual_encoder.num_features)

        return visual_encoder, ln_vision_qa, ln_vision_localizer 
    
    def init_t5_model(self, t5_model="google/flan-t5-xl") -> Union[
        T5ForConditionalGeneration, T5TokenizerFast
    ]:
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        return T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config), T5TokenizerFast.from_pretrained(t5_model)
    
    def init_Qformer(self, t5_model, num_query_token, vision_width):
        Qformer, query_tokens = super().init_Qformer(num_query_token=num_query_token, vision_width=vision_width)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        projection = nn.Linear(Qformer.config.hidden_size, t5_model.config.hidden_size)

        return Qformer, projection, query_tokens
    
    def repeat_frame_prefix(self, num_frames):
        return [SEVILA_MULTI_FRAME_PREFIX.format(str(i + 1)) for i in range(num_frames)]

    def get_visual_prefix_tokens(self, visual_tokens, B=None, T=None):
        """
        Obtains the prefix tokens for visual frames, prior to input to the T5 model.
        """
        # Use single frame prefix is both B and T is provided, else multi frame prefix
        frame_prefix = self.frame_prefix if T is not None else self.frame_seq_prefix

        frame_prefix = self.t5_tokenizer(
            frame_prefix, padding="longest",
            add_special_tokens=False, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        ).to(visual_tokens.device) # (1, Nfp) if single, else (Nkf, Nfp)
        # B * T -> Localizer refinement, use each frame as separate visual
        if T is not None:
            frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, B*T, 0)
            frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, B*T, 0)
        # B -> Answerer QA fine-tuning, get B prefixes for B questions-answer pairs
        else:
            frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids.unsqueeze(0), B, 0)
            frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask.unsqueeze(0), B, 0)

        return frame_prefix_id, frame_prefix_mask

    def get_text_tokens(self, input_text, device, T=None):
        input_tokens = self.t5_tokenizer(
            input_text, padding="longest", 
            truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        ).to(device)
        # For localizer pseudo-labeling: We repeat the QA prompt T times for each frame 
        # to determine if that frame can be used to answer the frame accurately.
        input_ids, attention_mask = input_tokens.input_ids, input_tokens.attention_mask
        if T is not None:
            input_ids = torch.repeat_interleave(input_ids, T, 0)
            attention_mask = torch.repeat_interleave(attention_mask, T, 0)

        return input_ids, attention_mask
    
    def get_query_tokens(
            self, 
            Qformer, query_tokens, projection, # Specific Qformer, query tokens and projection layer, for different task encoders (answerer, localizer)
            visual_tokens, visual_attention_mask
        ):
        """
        Given visual tokens extracted from a visual encoder(ViT) and learnable task-specific query tokens,
        Obtain visual features for the T5 model using a QFormer and a projection layer.
        Acts as a form of visual prompt generator/translator.
        """
        query_embeds = query_tokens.expand(visual_tokens.shape[0], -1, -1)
        outputs = Qformer.bert(
            query_embeds=query_embeds,
            encoder_hidden_states=visual_tokens,
            encoder_attention_mask=visual_attention_mask,
            return_dict=True
        )
        outputs = projection(outputs.last_hidden_state)
        attention_mask = torch.ones(outputs.shape[:-1], dtype=torch.long).to(visual_tokens.device)

        return outputs, attention_mask
    
    def get_input_output_embeddings(self,
        visual_tokens, visual_attention_mask,
        input_text, output_text,
        Qformer, query_tokens, projection,
        video_shape
    ):
        """
        Constructs the overall prompt to the T5 model, using the frame prefixes, query tokens and text inputs.
        Prompt format is ["Frame: ", <visual_features>, <prompt_features>]
        """
        B, T, C, H, W = video_shape

        # Single frame level visual embeddings - (B * T, N, D) (E.g. For localizer refinement/selection)
        if len(visual_tokens.shape) < 4:
            frame_prefix_id, frame_prefix_mask = self.get_visual_prefix_tokens(visual_tokens=visual_tokens, B=B, T=T) # (B*T, Nfp, D)
            query_tokens, query_tokens_mask = self.get_query_tokens(
                Qformer, query_tokens, projection,
                visual_tokens, visual_attention_mask
            ) # (B*T, Nq, D)

            # Stack frame prefix and query tokens together
            query_tokens = torch.cat([self.t5_model.encoder.embed_tokens(frame_prefix_id), query_tokens], dim=1)
            query_tokens_mask = torch.cat([frame_prefix_mask, query_tokens_mask], dim=1) 

            # Repeat T times for each frame, since each frame used in separate queries
            input_ids, input_attention_mask = self.get_text_tokens(input_text, device=visual_tokens.device, T=T) # (B*T, Nt)

            if output_text is not None:
                T = T if len(output_text) < visual_tokens.shape[0] else None # Repeat for answerer text during localizer refinement
                output_ids, output_attention_mask = self.get_text_tokens(output_text, device=visual_tokens.device, T=T)
            else:
                output_ids, output_attention_mask = None, None
       
        # Frame sequence visual embeddings - (B, T, N, D) (E.g. for QA fine-tuning/inference)
        # Here T refers to the number of keyframes
        else:
            frame_prefix_id, frame_prefix_mask = self.get_visual_prefix_tokens(
                visual_tokens=visual_tokens, B=B
            ) # (B, T, Nf)

            # Stack back to (B*T, N, D) for query token extraction
            B, T, _, _ = visual_tokens.shape
            visual_tokens = rearrange(visual_tokens, "b t n d -> (b t) n d")
            visual_attention_mask = rearrange(visual_attention_mask, "b t n -> (b t) n")

            query_tokens, query_tokens_mask = self.get_query_tokens(
                Qformer, query_tokens, projection,
                visual_tokens, visual_attention_mask
            ) 
            # Reshape back to (B, T, Nq, D)
            query_tokens = rearrange(query_tokens, "(b t) n d -> b t n d", b=B, t=T)
            query_tokens_mask = rearrange(query_tokens_mask, "(b t) n -> b t n", b=B, t=T)

            # Stack prefixes and query tokens together, in alternating respective frame prefix and query token
            query_tokens = rearrange(
                torch.cat([
                    self.t5_model.encoder.embed_tokens(frame_prefix_id),
                    query_tokens
                ], dim=2), # (B, T, Nq + Nf, D)
                "b t n d -> b (t n) d"
            )
            query_tokens_mask = rearrange(
                torch.cat([frame_prefix_mask, query_tokens_mask], dim=2), # (B, T, Nq + Nf)
                "b t n -> b (t n)"
            )

            input_ids, input_attention_mask = self.get_text_tokens(input_text, device=visual_tokens.device)

            if output_text is not None:
                output_ids, output_attention_mask = self.get_text_tokens(output_text, device=visual_tokens.device)
            else:
                output_ids, output_attention_mask = None, None

        if output_ids is not None:
            output_ids = output_ids.masked_fill(output_ids == self.t5_tokenizer.pad_token_id, -100)

        # Construct overall localizer prompt
        input_embeds = torch.cat([query_tokens, self.t5_model.encoder.embed_tokens(input_ids)], dim=1) # (B, T', D)
        attention_mask = torch.cat([query_tokens_mask, input_attention_mask], dim=1) # (B, T')

        return input_embeds, attention_mask, output_ids, output_attention_mask
    
    def generate_t5(self, 
        visual_tokens, visual_attention_mask,
        input_text, output_text,
        Qformer, query_tokens, projection,
        video_shape,
        # Generation hyperparameters
        do_sample=False,
        top_p=0.9,
        temperature=1.,
        max_new_tokens=30,
        min_length=1,
        num_beams=1,
        repetition_penalty=1.,
        length_penalty=1.,
        num_return_sequences=1
    ):
        """
        Generates answers from the T5 model, constructing the necessary tokens using
        respective specific models as needed.

        Args:
            visual_tokens, visual_attention_mask: Visual tokens and corresponding attention mask extracted from ViT
            input_text: Text prompt; Can be either answerer (question) or localizer text prompt
            output_text: Answers to accompanying questions for answerer, or pseudo-labels for localizer

        Returns:
            outputs: Output of the localizer T5 model.
        """
        # Construct input prompt and output labels for T5 model
        input_embeds, attention_mask, _, _ = self.get_input_output_embeddings(
            visual_tokens, visual_attention_mask,
            input_text, output_text,
            Qformer, query_tokens, projection,
            video_shape
        )

        outputs = self.t5_model.generate(
            inputs_embeds=input_embeds, attention_mask=attention_mask,
            do_sample=do_sample, top_p=top_p, temperature=temperature,
            max_new_tokens=max_new_tokens, min_length=min_length, 
            repetition_penalty=repetition_penalty, length_penalty=length_penalty, 
            num_return_sequences=num_return_sequences, num_beams=num_beams,
            return_dict_in_generate=True, output_hidden_states=True, output_scores=True
        )

        return outputs

    def forward_t5(self, 
        visual_tokens, visual_attention_mask,
        input_text, output_text,
        Qformer, query_tokens, projection,
        video_shape
    ):
        """
        Performs a forward function through the T5 model, constructing the necessary tokens using
        respective specific models as needed.

        Args:
            visual_tokens, visual_attention_mask: Visual tokens and corresponding attention mask extracted from ViT
            input_text: Text prompt; Can be either answerer (question) or localizer text prompt
            output_text: Answers to accompanying questions for answerer, or pseudo-labels for localizer

        Returns:
            outputs: Output of the localizer T5 model.
        """
        # Construct input prompt and output labels for T5 model
        input_embeds, attention_mask, output_ids, output_attention_mask = self.get_input_output_embeddings(
            visual_tokens, visual_attention_mask,
            input_text, output_text,
            Qformer, query_tokens, projection,
            video_shape
        )

        # Generate outputs from T5 model
        outputs = self.t5_model.forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=output_ids,
            decoder_attention_mask=output_attention_mask,
            return_dict=True
        )

        return outputs

    def forward(self, samples,
        do_sample=False,
        num_beams=1, 
        max_new_tokens=30,
        min_length=1, 
        top_p=0.9,
        repetition_penalty=1.0, 
        length_penalty=1.0,
        num_return_sequences=1, 
        temperature=1,):

        video = samples["video"]

        # Extract visual tokens
        B, T, C, W, H = video.shape     
        video = rearrange(video, "b t c h w -> (b t) c h w")
        visual_tokens = self.visual_encoder(video) 
        visual_attention_mask = torch.ones(visual_tokens.size()[:-1], dtype=torch.long).to(video.device) # bt n c

        answerer_input_text, localizer_input_text, answer_text = samples['qa_input'], samples['loc_input'], samples['qa_output']
        
        # Localizer self-refinement
        if 'train_loc' in self.task:
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
        
        # Finetune answerer with localizer
        elif 'train_qa_with_loc' in self.task:
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
                loss = outputs.loss
                
        return {
            "loss" : loss
        }

    @torch.no_grad()
    def predict_answers(self,
        samples,
        do_sample=False,
        num_beams=1, max_new_tokens=30,
        min_length=1, top_p=0.9,
        repetition_penalty=1.0, length_penalty=1.0,
        num_return_sequences=1, temperature=1,):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        video, qid = samples["video"], samples['question_id']
        answerer_input_text, localizer_input_text = samples['qa_input'], samples['loc_input']
        answer = samples['qa_output'] if 'qa_output' in samples else None

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            B, T, C, W, H = video.shape        
            video = rearrange(video, "b t c h w -> (b t) c h w")
            visual_tokens = self.visual_encoder(video) # bt, n, c
            topk_attention_mask = torch.ones(visual_tokens.size()[:-1], dtype=torch.long).to(video.device)

            outputs_loc = self.generate_t5(
                self.ln_vision_localizer(visual_tokens.detach().clone()), topk_attention_mask.detach().clone(),
                localizer_input_text, None, # No output labels for localizer when generating
                self.Qformer_localizer, self.query_tokens_localizer, self.projection_localizer,
                video_shape=(B, T, C, H, W),
                do_sample=do_sample, top_p=top_p, temperature=temperature,
                max_new_tokens=max_new_tokens, min_length=min_length, 
                repetition_penalty=repetition_penalty, length_penalty=length_penalty, 
                num_return_sequences=num_return_sequences, num_beams=num_beams
            )
                    
            localizer_logits = outputs_loc.scores[0]
            positive_logits = localizer_logits[:, self.id_positive].reshape(B, -1)

            visual_tokens = self.ln_vision_answerer(visual_tokens)
            # Select top K frames
            topk_idx = torch.topk(positive_logits, self.num_keyframes, dim=-1).indices
            topk_idx = torch.tensor([sorted(idx.tolist()) for idx in topk_idx]).to(visual_tokens.device)
            visual_tokens = rearrange(visual_tokens, "(b t) n d -> b t n d", b=B, t=T)
            topk_visual_tokens = torch.stack([visual_tokens[i, row, :, :] for i, row in enumerate(topk_idx)])
            topk_attention_mask = torch.ones(topk_visual_tokens.shape[:-1], dtype=torch.long).to(video.device)

            outputs = self.generate_t5(
                topk_visual_tokens, topk_attention_mask,
                answerer_input_text, None,
                self.Qformer_answerer, self.query_tokens_answerer, self.projection_answerer,
                video_shape=(B, T, C, H, W),
                do_sample=do_sample, top_p=top_p, temperature=temperature,
                max_new_tokens=max_new_tokens, min_length=min_length, 
                repetition_penalty=repetition_penalty, length_penalty=length_penalty, 
                num_return_sequences=num_return_sequences, num_beams=num_beams
            )
            logits = outputs.scores[1][:, self.answer_ids]
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
    
        return {
            'output_text' : preds,
            'answer' : answer,
            'qid' : qid
        }

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        frame_num = cfg.get("frame_num", 8)
        answer_num = cfg.get("answer_num", 5) 
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        task = cfg.get("task", 'train_loc_freeze_qa')

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            frame_num=frame_num,
            answer_num=answer_num,
            task=task,
        )
        model.load_checkpoint_from_config(cfg)
        # for sevila with qvh pretraining
        # need load blip-2 q-former ckpt to q-former_loc
        if 'loc' in task and 'qvh' not in task:
           model.load_qformer_loc()

        return model