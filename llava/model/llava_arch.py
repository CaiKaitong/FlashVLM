#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config, fastv_config=None):
        if fastv_config is not None:
            super(LlavaMetaModel, self).__init__(config, fastv_config)
        else:
            super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # VisPruner
    # def encode_images(self, images):
    #     image_features, image_attentions = self.get_model().get_vision_tower()(images, output_attentions=True) # (B, N, C), (B, M, N)

    #     features_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, C)
    #     image_attentions = image_attentions.mean(dim=1) # (B, N)

    #     B, N = image_features.shape[:2]
    #     visual_token_num = self.get_visual_token_num() # T
    #     important_token_num = int(visual_token_num * 0.5) # T_imp
    #     diverse_token_num = visual_token_num - important_token_num # T_div

    #     token_indices = image_attentions.argsort(dim=-1, descending=True) # (B, N)
    #     important_indices = token_indices[:, :important_token_num] # (B, T_imp)
    #     residual_indices = token_indices[:, important_token_num:] # (B, N - T_imp)

    #     while True:
    #         residual_tokens = features_normalized[torch.arange(B), residual_indices] # (B, R, C)
    #         r = min(8, residual_tokens.shape[1] - diverse_token_num)
    #         if r <= 0:
    #             break

    #         a, b = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :] # (B, R // 2, C)
    #         scores = a @ b.transpose(-1, -2) # (B, R // 2, R // 2)
    #         scores = scores.max(dim=-1).values # (B, R // 2)

    #         distinct_indices = scores.argsort(dim=-1, descending=True)[:, r:] # (B, R // 2 - r)
    #         residual_indices = torch.cat([residual_indices[..., ::2][torch.arange(B), distinct_indices], residual_indices[..., 1::2]], dim=-1) # (B, R - r)

    #     token_indices = torch.cat([important_indices, residual_indices], dim=-1) # (B, T)
    #     token_indices = torch.sort(token_indices).values # (B, T)
    #     image_features = image_features[torch.arange(B), token_indices]

    #     image_features = self.get_model().mm_projector(image_features)
    #     return image_features, image_features.shape[1]

    # # VisPruner
    # def prepare_inputs_labels_for_multimodal(
    #     self, input_ids, position_ids, attention_mask, past_key_values, labels,
    #     images, modalities=["image"], image_sizes=None
    # ):
    #     vision_tower = self.get_vision_tower()
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         return input_ids, position_ids, attention_mask, past_key_values, None, labels

    #     if type(images) is list or images.ndim == 5:
    #         if type(images) is list:
    #             images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
    #         concat_images = torch.cat([image for image in images], dim=0)
    #         image_features = self.encode_images(concat_images)
    #         split_sizes = [image.shape[0] for image in images]
    #         image_features = torch.split(image_features, split_sizes, dim=0)
    #         mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
    #         image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
    #         if mm_patch_merge_type == 'flat':
    #             image_features = [x.flatten(0, 1) for x in image_features]
    #         elif mm_patch_merge_type.startswith('spatial'):
    #             new_image_features = []
    #             for image_idx, image_feature in enumerate(image_features):
    #                 if image_feature.shape[0] > 1:
    #                     base_image_feature = image_feature[0]
    #                     image_feature = image_feature[1:]
    #                     height = width = self.get_vision_tower().num_patches_per_side
    #                     assert height * width == base_image_feature.shape[0]
    #                     if image_aspect_ratio == 'anyres':
    #                         num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
    #                         image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
    #                     else:
    #                         raise NotImplementedError
    #                     if 'unpad' in mm_patch_merge_type:
    #                         image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
    #                         image_feature = image_feature.flatten(1, 2).flatten(2, 3)
    #                         image_feature = unpad_image(image_feature, image_sizes[image_idx])
    #                         image_feature = torch.cat((
    #                             image_feature,
    #                             self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
    #                         ), dim=-1)
    #                         image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    #                     else:
    #                         image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
    #                         image_feature = image_feature.flatten(0, 3)
    #                     image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    #                 else:
    #                     image_feature = image_feature[0]
    #                     if 'unpad' in mm_patch_merge_type:
    #                         image_feature = torch.cat((
    #                             image_feature,
    #                             self.model.image_newline[None].to(image_feature.device)
    #                         ), dim=0)
    #                 new_image_features.append(image_feature)
    #             image_features = new_image_features
    #         else:
    #             raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    #     else:
    #         image_features, visual_token_num = self.encode_images(images)

    #     # TODO: image start / end is not implemented here to support pretraining.
    #     if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #         raise NotImplementedError

    #     # Let's just add dummy tensors if they do not exist,
    #     # it is a headache to deal with None all the time.
    #     # But it is not ideal, and if you have a better idea,
    #     # please open an issue / submit a PR, thanks.
    #     _labels = labels
    #     _position_ids = position_ids
    #     _attention_mask = attention_mask
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    #     else:
    #         attention_mask = attention_mask.bool()
    #     if position_ids is None:
    #         position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    #     if labels is None:
    #         labels = torch.full_like(input_ids, IGNORE_INDEX)

    #     # remove the padding using attention_mask -- FIXME
    #     _input_ids = input_ids
    #     input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    #     labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    #     new_input_embeds = []
    #     new_labels = []
    #     cur_image_idx = 0
    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    #         if num_images == 0:
    #             cur_image_features = image_features[cur_image_idx]
    #             cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
    #             new_input_embeds.append(cur_input_embeds)
    #             new_labels.append(labels[batch_idx])
    #             cur_image_idx += 1
    #             continue

    #         image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    #         cur_input_ids_noim = []
    #         cur_labels = labels[batch_idx]
    #         cur_labels_noim = []
    #         for i in range(len(image_token_indices) - 1):
    #             cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
    #             cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
    #         split_sizes = [x.shape[0] for x in cur_labels_noim]
    #         cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
    #         cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    #         cur_new_input_embeds = []
    #         cur_new_labels = []

    #         for i in range(num_images + 1):
    #             cur_new_input_embeds.append(cur_input_embeds_no_im[i])
    #             cur_new_labels.append(cur_labels_noim[i])
    #             if i < num_images:
    #                 cur_image_features = image_features[cur_image_idx]
    #                 cur_image_idx += 1
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

    #         cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    #         cur_new_labels = torch.cat(cur_new_labels)

    #         new_input_embeds.append(cur_new_input_embeds)
    #         new_labels.append(cur_new_labels)

    #     # Truncate sequences to max length as image embeddings can make the sequence longer
    #     tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    #     if tokenizer_model_max_length is not None:
    #         new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    #         new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    #     # Combine them
    #     max_len = max(x.shape[0] for x in new_input_embeds)
    #     batch_size = len(new_input_embeds)

    #     new_input_embeds_padded = []
    #     new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    #     attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    #     position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    #     for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
    #         cur_len = cur_new_embed.shape[0]
    #         if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
    #             new_input_embeds_padded.append(torch.cat((
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
    #                 cur_new_embed
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, -cur_len:] = cur_new_labels
    #                 attention_mask[i, -cur_len:] = True
    #                 position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
    #         else:
    #             new_input_embeds_padded.append(torch.cat((
    #                 cur_new_embed,
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, :cur_len] = cur_new_labels
    #                 attention_mask[i, :cur_len] = True
    #                 position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    #     new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    #     if _labels is None:
    #         new_labels = None
    #     else:
    #         new_labels = new_labels_padded

    #     if _attention_mask is None:
    #         attention_mask = None
    #     else:
    #         attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    #     if _position_ids is None:
    #         position_ids = None

    #     return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, visual_token_num




    # FlashVLM
    def encode_images(
        self,
        images,
        text_embeds=None,
        *,
        eta: float = 0.5,
    ):
        import os
        import torch
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        image_features, image_attentions = self.get_model().get_vision_tower()(
            images, output_attentions=True
        )
        features_normalized = F.normalize(image_features, dim=-1)
        attn = image_attentions.mean(dim=1)  # (B, N)

        B, N = image_features.shape[:2]
        T = self.get_visual_token_num()
        T_imp = int(T * 0.5)
        T_div = T - T_imp

        # 2) 文本引导相似度
        if text_embeds is not None:
            with torch.no_grad():
                img_lm = self.get_model().mm_projector(image_features)  # (B, N, D_lm)
                img_lm = F.normalize(img_lm, dim=-1)

            text_sim_list = []
            for b in range(B):
                tb = text_embeds[b] if isinstance(text_embeds, (list, tuple)) else text_embeds
                if tb is None:
                    print("error")
                    text_sim_list.append(torch.zeros(N, device=img_lm.device, dtype=img_lm.dtype))
                    continue
                # --- 维度对齐 ---
                tb = F.normalize(tb, dim=-1)
                D_lm = img_lm.shape[-1]
                if tb.shape[-1] != D_lm:
                    if tb.shape[-1] < D_lm:
                        tb = F.pad(tb, (0, D_lm - tb.shape[-1]))
                    else:
                        tb = tb[..., :D_lm]
                # --- 🔸Norm-based gating: 强调语义词 (如名词)，抑制停用词 ---
                text_norm = tb.norm(dim=-1)
                tb = tb * (text_norm / (text_norm.max() + 1e-6)).unsqueeze(-1)
                # --- 多 token 语义方向 ---
                sim_mat = torch.matmul(img_lm[b], tb.transpose(0, 1))  # (N, L_text)
                sim = sim_mat.max(dim=-1).values                        # (N,)

                weights = F.softmax(sim_mat / 0.05, dim=-1)
                sim = (sim_mat * weights).sum(dim=-1)


                # --- 对比度超强化 ---
                T = 0.010   # 温度越小越尖锐，可在 0.02–0.05 之间调
                sim = F.softmax(sim / T, dim=0)

                # 幂放大：提升亮区、压暗区
                gamma = 2.5   # 越大对比越强，可调到 2.5
                sim = sim ** gamma

                # 再归一化到 [0,1]
                sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)

                # --- Top-p 截断（保留最亮部分，暗区极暗）---
                top_p = 0.005   # 仅保留最亮 10%，其余抑制
                sim = sim.float()  # ✅ 确保为 float32
                thresh = torch.quantile(sim, 1 - top_p)
                sim = torch.where(sim >= thresh, sim, sim * 0.1)

                # --- 局部对比再平衡（保持亮区细节）---
                sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)

                text_sim_list.append(sim)

            text_sim = torch.stack(text_sim_list, dim=0)
        else:
            text_sim = torch.zeros_like(attn)

        # 3) 融合分数
        attn_min = attn.amin(dim=-1, keepdim=True)
        attn_max = attn.amax(dim=-1, keepdim=True)
        attn_norm = (attn - attn_min) / (attn_max - attn_min + 1e-6)

        # 避免 0 值
        eps = 1e-6
        A = attn_norm.clamp(min=eps)
        T = text_sim.clamp(min=eps)

        # 对数域加权
        score = torch.exp((1 - eta) * torch.log(A) + eta * torch.log(T))

        # 再归一化
        score = (score - score.min(dim=-1, keepdim=True).values) / (
            score.max(dim=-1, keepdim=True).values - score.min(dim=-1, keepdim=True).values + 1e-6
        )

        token_indices = score.argsort(dim=-1, descending=True)
        important_indices = token_indices[:, :T_imp]
        residual_indices  = token_indices[:, T_imp:]

        while True:
            residual_tokens = features_normalized[torch.arange(B).unsqueeze(-1), residual_indices]
            r = min(8, residual_tokens.shape[1] - T_div)
            if r <= 0:
                break
            a, b = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :]
            scores = torch.matmul(a, b.transpose(-1, -2)).max(dim=-1).values
            distinct_indices = scores.argsort(dim=-1, descending=True)[:, r:]
            even_idx = residual_indices[..., ::2]
            odd_idx  = residual_indices[..., 1::2]
            kept_even = even_idx[torch.arange(B).unsqueeze(-1), distinct_indices]
            residual_indices = torch.cat([kept_even, odd_idx], dim=-1)

        token_indices = torch.cat([important_indices, residual_indices], dim=-1)
        token_indices = torch.sort(token_indices, dim=-1).values

        image_features = image_features[torch.arange(B).unsqueeze(-1), token_indices]
        image_features = self.get_model().mm_projector(image_features)

        return image_features, image_features.shape[1]





    # FlashVLM
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, modalities=["image"], image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # ---------------------------------------------------------------------
        # 基本准备（保持原样）
        # ---------------------------------------------------------------------
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # ---------------------------------------------------------------------
        # === CHANGE (1): 先去 padding，再提取每个样本的“文本中心向量”
        # ---------------------------------------------------------------------
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # 统计每个样本中 <image> 出现次数，并计算“文本中心向量”（全局平均，稳妥）
        per_sample_num_images = []
        per_sample_text_center = []  # [None] 或 [1, D] 的张量
        for cur_input_ids in input_ids:
            # <image> token 计数
            image_token_pos = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            num_images = int(image_token_pos.numel())
            per_sample_num_images.append(num_images)

            # 文本 token（把 <image> 去掉）
            # 注：保持与后续拼接逻辑一致的拆分方式
            image_token_indices = [-1] + image_token_pos.tolist() + [cur_input_ids.shape[0]]
            text_spans = []
            for i in range(len(image_token_indices) - 1):
                text_spans.append(cur_input_ids[image_token_indices[i] + 1: image_token_indices[i + 1]])

            if len(text_spans) > 0:
                split_sizes = [x.shape[0] for x in text_spans if x.numel() > 0]
                if len(split_sizes) > 0:
                    flat_ids = torch.cat([x for x in text_spans if x.numel() > 0], dim=0)
                    flat_emb = self.get_model().embed_tokens(flat_ids)  # [L, D]
                    # 不再求 mean，而是保留完整的文本序列
                    text_center = flat_emb                              # [L, D]
                    per_sample_text_center.append(text_center)
                else:
                    per_sample_text_center.append(None)
            else:
                per_sample_text_center.append(None)



        # ---------------------------------------------------------------------
        # === CHANGE (2): 再去编码图像，并把“文本中心向量”传入 encode_images
        #     需要让 encode_images 支持参数 text_embeds（列表），
        #     用于在视觉侧做文本引导（如 cosine 重标定 + VisPruner筛选）
        # ---------------------------------------------------------------------
        if type(images) is list or images.ndim == 5:
            # list/5D 分支：原逻辑按列表顺序 cat；我们需要把 text_embeds 展平成与 concat_images 对齐的列表
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)

            # 构造与 concat_images 同长度的 text_embeds 列表。
            # 假设 images 按 batch 样本顺序与 <image> 出现顺序一致（与后续 cur_image_idx 一致）。
            # 对于每个样本，若该样本有 num_images 次 <image>，重复该样本的 text_center num_images 次。
            text_embeds_for_concat = []
            for text_center, num_images in zip(per_sample_text_center, per_sample_num_images):
                repeat_n = max(1, num_images) if type(images) is list else num_images
                if repeat_n == 0:
                    # 极端：若没有 <image>，该样本不会消耗 image_features，不需要占位
                    continue
                for _ in range(repeat_n):
                    text_embeds_for_concat.append(text_center)  # 允许 None

            # === 这里调用你“改造后的” encode_images ===
            # 你需要给 encode_images 增加一个缺省参数 text_embeds=None
            image_features = self.encode_images(concat_images, text_embeds=text_embeds_for_concat)

            # 拆回原列表形状
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)

            # 原 mm_patch_merge_type 分支保持不变
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx],
                                self.config.image_grid_pinpoints,
                                self.get_vision_tower().config.image_size
                            )
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

            visual_token_num = None  # list 分支原来没有返回 visual_token_num，这里保持一致

        else:
            # 普通张量分支：B 张图像，与 batch 对齐；给每个样本一个 text_center
            text_embeds_batch = [tc for tc in per_sample_text_center]  # len == B
            image_features, visual_token_num = self.encode_images(images, text_embeds=text_embeds_batch)

        # ---------------------------------------------------------------------
        # 预训练相关限制（保持原样）
        # ---------------------------------------------------------------------
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # ---------------------------------------------------------------------
        # 下面开始：拼接 text + image（原逻辑基本不动）
        # ---------------------------------------------------------------------
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # 这里原代码会用 image_features[cur_image_idx]，但没有图像时我们不消耗 image_features
                # 保持原逻辑：嵌入文本 + 空的图像特征
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, image_features[cur_image_idx][0:0] if (type(images) is list or images.ndim == 5) else image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                # 注意：原代码这里有 cur_image_idx += 1，但当 num_images==0 不应前进；这里修正不挪动
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    # 这里按编码顺序取出对应的 image_features（顺序与上面 text_embeds_for_concat 保持一致）
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # 截断到 tokenizer 上限（保持原样）
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # 对齐 batch 尺寸（保持原样）
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, visual_token_num





    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
