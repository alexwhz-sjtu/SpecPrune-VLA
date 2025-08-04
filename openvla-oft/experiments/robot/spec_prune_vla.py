import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.util import view_as_blocks


@torch.no_grad()

def patchify(image, patch_size=14):
    """
    Converts an image into non-overlapping patches (PyTorch GPU version).
    Args:
        image: Input image (PIL or numpy array).
        patch_size: Size of each patch (default=14).
        device: Target device ('cuda' or 'cpu').
    Returns:
        Patches as a PyTorch tensor on GPU.
    """
    image = np.array(image)
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0, "Image dimensions must be divisible by patch size."

    if image.ndim == 3:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size, image.shape[2]))
    else:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size))

    patches = blocks.reshape(-1, patch_size, patch_size, image.shape[2]) if image.ndim == 3 else blocks.reshape(-1, patch_size, patch_size)
    return patches

def calculate_patch_similarity(patch1, patch2):
    """
    Computes cosine similarity between two sets of patches.
    """
    flat1 = patch1.reshape(len(patch1), -1).astype(np.float32)
    flat2 = patch2.reshape(len(patch2), -1).astype(np.float32)
    
    norm1 = np.linalg.norm(flat1, axis=1)
    norm2 = np.linalg.norm(flat2, axis=1)
    
    dot = np.sum(flat1 * flat2, axis=1)
    cosine_sim = dot / (norm1 * norm2 + 1e-8)
    return cosine_sim

def find_static_patches(img_0, img_1, patch_size=14, top_k=50, sim_threshold=0.999):
    """
    Identifies significant patches with high similarity across two images.
    """
    patches1 = patchify(img_0, patch_size)
    patches2 = patchify(img_1, patch_size)

    similarity = calculate_patch_similarity(patches1, patches2)
    grid_size = 224 // patch_size
    similarity_2d = similarity.reshape(grid_size, grid_size)

    patch_scores = [(i * grid_size + j, similarity_2d[i, j])
                    for i in range(grid_size) for j in range(grid_size)
                    if similarity_2d[i, j] >= sim_threshold]

    patch_scores.sort(key=lambda x: x[1], reverse=True)
    top_patch_ids = [idx for idx, _ in patch_scores[:80]]

    return top_patch_ids

def draw_patches_overlay(image, patch_groups, patch_size=14, alpha=0.4, border_width=2):
    """
    在图像上绘制带有边框的彩色覆盖层。
    
    Args:
        image: 输入图像 (PIL Image)
        patch_groups: 补丁组列表，每组包含 (patch_ids, color)
        patch_size: 每个补丁的大小
        alpha: 填充颜色的透明度
        border_width: 边框宽度
    """
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width = image.size[0]
    num_patches = width // patch_size

    for patch_list, color in patch_groups:
        # 转换为RGB格式，用于调整边框颜色
        border_color = (
            min(int(color[0] * 1.2), 255),  # 边框颜色稍微比填充色深一些
            min(int(color[1] * 1.2), 255),
            min(int(color[2] * 1.2), 255),
            255
        )
        
        for pid in patch_list:
            i, j = divmod(pid, num_patches)
            top_left = (j * patch_size, i * patch_size)
            bottom_right = ((j + 1) * patch_size, (i + 1) * patch_size)
            
            # 绘制填充
            draw.rectangle([top_left, bottom_right], 
                         fill=color + (int(255 * alpha),))
            
            # # 绘制边框
            # draw.rectangle([top_left, bottom_right], 
            #              outline=border_color,
            #              width=border_width)

    return Image.alpha_composite(image, overlay).convert("RGB")


def visualize_sim_patches(img_0, img_1, top_k):
    high_sim_patch_ids = find_static_patches(img_0, img_1, patch_size=14, top_k=top_k, sim_threshold=0.999)
    patch_groups = [(high_sim_patch_ids, (241, 196, 15))]

    result_image = draw_patches_overlay(img_1, patch_groups, patch_size=14, alpha=0.4)
    return np.array(result_image)

def visualize_similarity_indices(img_0, img_1, frame_idx, total_episodes, top_k, patch_size=14, sim_threshold=0.996, view="primary"):
    patches1 = patchify(img_0, patch_size)
    patches2 = patchify(img_1, patch_size)

    similarity = calculate_patch_similarity(patches1, patches2)
    grid_size = 224 // patch_size
    similarity_2d = similarity.reshape(grid_size, grid_size)

    patch_scores = [(i * grid_size + j, similarity_2d[i, j])
                    for i in range(grid_size) for j in range(grid_size)
                    if similarity_2d[i, j] >= sim_threshold]

    patch_scores.sort(key=lambda x: x[1], reverse=True)
    patcher_to_draw = [idx for idx, _ in patch_scores[:min(top_k, len(patch_scores)-1)]]
    patch_groups = [(patcher_to_draw, (241, 196, 15))]

    result_image = draw_patches_overlay(img_1, patch_groups, patch_size=14, alpha=0.4)
    from experiments.robot.libero.libero_utils import (
            save_rollout_frame,
        )
    dir = save_rollout_frame(result_image, 40, frame_idx, total_episodes, log_file=None, view=view)
    return patcher_to_draw

import numpy as np

def get_similarity_indices(img_0, img_1, top_k, patch_size=14, sim_threshold=0.996, view="primary"):
    # Step 1: Patchify both images
    patches1 = patchify(img_0, patch_size)
    patches2 = patchify(img_1, patch_size)

    # Step 2: Calculate similarity (assume cosine or pearson correlation)
    similarity = calculate_patch_similarity(patches1, patches2)

    # Step 3: Filter and select top-k indices
    grid_size = 224 // patch_size
    flat_indices = np.where(similarity >= sim_threshold)[0]  # flat indices of high-sim patches
    
    if len(flat_indices) == 0:
        return np.array([], dtype=int)

    # Use argpartition to get top-k most similar patches
    k = min(top_k, len(flat_indices))
    top_k_flat = flat_indices[np.argpartition(similarity[flat_indices], -k)[-k:]]

    # Convert flat indices to 2D coordinates and then to patch IDs
    row_ids, col_ids = np.unravel_index(top_k_flat, (grid_size, grid_size))
    patch_ids = row_ids * grid_size + col_ids + 1  # +1 for 1-based indexing

    # Add offset for wrist view
    if view == "wrist":
        patch_ids += 256

    return patch_ids

def vlm_layer_attn(multihead_attention, num_tokens=34, layer_indices=None, primary=True):
    # multihead_attention (tuple) --> (attention[layer_idx], cache_position[layer_idx])
    num_layers = len(multihead_attention)

    layer_indices = range(layer_indices) if layer_indices is not None else range(num_layers)
   
    v_token_start = 1 if primary else 257
    v_token_end = v_token_start + 256
    t_token_start = 513
    t_token_end = t_token_start + 34
    
    # 获取注意力位置
    # attention_pos = multihead_attention[-1]
    # token_ids = torch.arange(700)
    # text_mask = (token_ids >= t_token_start) & (token_ids < t_token_end)
   
    # 初始化结果字典
    attn_dict = {}
    token_position = {}
    num_tokens = 0
    for layer_idx in layer_indices:
        attn_dict[layer_idx] = {}
        attn_map = multihead_attention[layer_idx][0].to(torch.float32).squeeze(0).mean(dim=0)
        token_idx = multihead_attention[layer_idx][1]

        # 使用实际的 token 数量来构造 mask

        text_mask = (token_idx >= t_token_start+9) & (token_idx < t_token_end)
        vision_mask = (token_idx >= v_token_start) & (token_idx < v_token_end)

        relation = attn_map[text_mask]  # [num_text_tokens, num_actual_vision_tokens]
        relation = relation[ : , vision_mask]
        token_position[layer_idx] = token_idx
        attn_dict[layer_idx] = relation.mean(dim=0)

    return attn_dict, token_position  

def visualize_layer_attn(img, multihead_attention, high_similarity_indices,  frame_idx, total_episodes, topk, primary=True):
    
    if primary :
        attn_dict, position= vlm_layer_attn(multihead_attention, layer_indices=None, primary=True)
    else:
        attn_dict, position = vlm_layer_attn(multihead_attention, layer_indices=None, primary=False)

    offset = 1 if primary else 257
    spec_indices = []
    blue_intensity = 100
    color = (46, 87, blue_intensity)
    total_patches = 256 + offset
    from experiments.robot.libero.libero_utils import (
        save_rollout_frame,
    )
    for layer_idx in attn_dict:
        if layer_idx in range(1,3):
            continue
        position_id = position[layer_idx]
        attn_scores = attn_dict[layer_idx]

        filtered_ids = position_id[(position_id >= offset) & (position_id < total_patches)] - offset
        
        full_attn = torch.zeros(256, device = attn_scores.device)

        full_attn[filtered_ids] = attn_scores
        
        full_attn = full_attn.cpu().numpy().astype(np.float32)

        attn = full_attn.reshape(16, 16)

        attn_resized = cv2.resize(attn, (16, 16))

        flat = [(i * 16 + j, attn_resized[i, j]) for i in range(16) for j in range(16)]
        flat.sort(key=lambda x: x[1], reverse=True)
        top_attn_patches = [idx for idx, _ in flat[:topk]]

        # color = "#2B5F28"
        top_patch_group = [(top_attn_patches, color)]

        if primary:
            dynamic_group = sorted(set(range(0,256)) - set(high_similarity_indices))
            final = list(set(dynamic_group) | set(top_attn_patches))
            final_group = [(final, color)]
            # result_image = (draw_patches_overlay(img, top_patch_group, patch_size=14, alpha=0.6))
            result_image = np.array(draw_patches_overlay(img, final_group, patch_size=14, alpha=0.6))
        else:
            dynamic_group_w = sorted(set(range(256,512)) - set(high_similarity_indices))
            dynamic_group_w = [idx-256 for idx in dynamic_group_w]
            final = list(set(dynamic_group_w) | set(top_attn_patches))
            final_group = [(final, color)]
            spec_group = [(top_attn_patches, color)]
            dynamic_group_w = [(dynamic_group_w,(45, 87, 150))]
            # result_image = draw_patches_overlay(img, dynamic_group_w, patch_size=14, alpha=0.6)
            # result_image = np.array(draw_patches_overlay(result_image, spec_group, patch_size=14, alpha=0.6))

            result_image = np.array(draw_patches_overlay(img, final_group, patch_size=14, alpha=0.6))



        if primary:
            dir = save_rollout_frame(result_image, layer_idx, frame_idx, total_episodes, log_file=None, view="primary")
            
        else:
            dir = save_rollout_frame(result_image, layer_idx , frame_idx, total_episodes, log_file=None, view="wrist")


    return torch.tensor(top_attn_patches) + offset



# # return {layer_idx: {head_idx: attention_scores}}
# def vlm_layer_attn(multihead_attention, layer_indices=None, primary=True):
#     """
#     Computes mean attention from text tokens to vision tokens for specified layers and heads.
    
#     Args:
#         multihead_attention: Tensor of shape (num_layers, num_heads, seq_len, seq_len) or similar.
#         layer_indices: List of layer indices to process. If None, process all layers.
#         primary: Boolean, determines vision token start index.
    
#     Returns:
#         dict: {layer_idx: {head_idx: attention_scores}}, where attention_scores is a tensor of shape (256,).
#     """
#     # 如果未指定层索引，处理所有层
#     # assert hasattr(multihead_attention, "shape"), f"multihead_attention type: {type(multihead_attention)}"
#     print([multihead_attention])
#     num_layers = len(multihead_attention)
#     print(num_layers)
#     layer_indices = layer_indices if layer_indices is not None else range(num_layers)
    
#     # 假设多头注意力张量的形状为 (num_layers, num_heads, seq_len, seq_len)
#     num_heads = multihead_attention[0].shape[1]
#     print(num_heads)
#     # 定义 token 范围，与 token_attention_merge 一致
#     v_token_start = 1 if primary else 257
#     v_token_end = v_token_start + 256
#     t_token_start = 513
#     t_token_end = t_token_start + 34
    
#     # 获取注意力位置
#     attention_pos = multihead_attention[-1]
#     
#     text_mask = (attention_pos_1d >= t_token_start) & (attention_pos_1d < t_token_end)
#     text_masks = text_mask[0]  # shape: [32, 599, 599]
#     # 初始化结果字典
#     result = {}
    
#     for layer_idx in layer_indices:
#         result[layer_idx] = {}
#         for head_idx in range(num_heads):
#             # 提取单层的单头注意力图，模仿 token_attention_merge
#             attn_map = multihead_attention[layer_idx][head_idx].to(torch.float32)
#             mask = text_masks[head].flatten()
#             # 提取文本到视觉的注意力子矩阵
#             relation = attn_map[text_mask, v_token_start:v_token_end]
            
#             # 计算text token dui1 img token 平均注意力，模仿 token_attention_merge
#             attention_scores = relation.mean(dim=0).cpu()
            
#             # 存储结果
#             result[layer_idx][head_idx] = attention_scores
    
#     return result