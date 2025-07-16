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


def draw_patches_overlay(image, patch_groups, patch_size=14, alpha=0.4):
    """
    Draws colored overlays on image for different patch groups.
    """
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width = image.size[0]
    num_patches = width // patch_size

    for patch_list, color in patch_groups:
        for pid in patch_list:
            i, j = divmod(pid, num_patches)
            top_left = (j * patch_size, i * patch_size)
            bottom_right = ((j + 1) * patch_size, (i + 1) * patch_size)
            draw.rectangle([top_left, bottom_right], fill=color + (int(255 * alpha),))

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

def vlm_layer_attn(multihead_attention, layer_indices=None, primary=True):

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
    result = {}
    num_tokens = 0
    for layer_idx in layer_indices:
        result[layer_idx] = {}
        attn_map = multihead_attention[layer_idx].to(torch.float32).squeeze(0).mean(dim=0)
        # 使用实际的 token 数量来构造 mask
        if not num_tokens:
            num_tokens = attn_map.size(0)  # 获取当前 attention map 的 token 数量
            token_ids = torch.arange(num_tokens)  
            text_mask = (token_ids >= t_token_start+7) & (token_ids < t_token_start+8)
            text_mask_wrist = (token_ids >= t_token_start) & (token_ids < t_token_end)
        if primary:
            relation = attn_map[text_mask, v_token_start:v_token_end]
        else:
            relation = attn_map[text_mask_wrist, v_token_start:v_token_end]

        result[layer_idx] = relation.mean(dim=0).cpu()

    return result

def visualize_layer_img_attn(img, multihead_attention, high_similarity_indices, frame_idx, total_episodes,  primary=True):
    if primary :
        attn_dict = vlm_layer_attn(multihead_attention, layer_indices=2, primary=True)
    else:
        attn_dict = vlm_layer_attn(multihead_attention, layer_indices=2, primary=False)
    patches_sum = set()
    for layer_idx in attn_dict:
        attn_scores = attn_dict[layer_idx]
        # 重塑为 16x16 网格，排序后返回 top-k 视觉 token 索引（默认 120）。
        attn_scores = attn_scores.cpu().numpy() if isinstance(attn_scores, torch.Tensor) else attn_scores
        attn = attn_scores.reshape(16, 16)
        attn_resized = cv2.resize(attn, (16, 16))

        flat = [(i * 16 + j, attn_resized[i, j]) for i in range(16) for j in range(16)]
        flat.sort(key=lambda x: x[1], reverse=True)
        top_attn_patches = [idx for idx, _ in flat[:60 if primary else 100]]
        top_attn_scores = [score for _, score in flat[:40]]
        overlap = []
        # 收集所有需要上色的 patch：top patch 所在列的所有 patch
        all_patches_to_highlight = set()
        if primary:
            for idx in top_attn_patches:
                col = idx % 16
                cur_row = idx // 16
                for row in range(max(0, (cur_row - 10)), cur_row+1):
                    patch_idx = row * 16 + col
                    all_patches_to_highlight.add(patch_idx)
                    if patch_idx+1 in high_similarity_indices:
                        overlap.append(patch_idx)
                    else:
                        patches_sum.add(patch_idx)

        else:
            topk_high_sim = set(high_similarity_indices[:20])
            high_sim_not_in_attn = set(high_similarity_indices) - set(top_attn_patches)
            patches_group = [
            (list(topk_high_sim), (15, 67, 50)),
            (list(high_sim_not_in_attn), (150,75,100))
        ]
        # 转换为 list
        all_patches_to_highlight = sorted(all_patches_to_highlight)

        blue_intensity = 50
        color = (15, 67, blue_intensity)
        top_patch_group = [(all_patches_to_highlight, color)]

        if primary:
            result_image = np.array(draw_patches_overlay(img, top_patch_group, patch_size=14, alpha=0.4))
        else:
            result_image = np.array(draw_patches_overlay(img, patches_group, patch_size=14, alpha=0.4))
       
        from experiments.robot.libero.libero_utils import (
            save_rollout_frame,
        )
        if primary:
            dir = save_rollout_frame(result_image, layer_idx, frame_idx, total_episodes, log_file=None, view="primary")
            
        else:
            dir = save_rollout_frame(result_image, layer_idx , frame_idx, total_episodes, log_file=None, view="wrist")

    if primary:
        patches_sum = sorted(patches_sum)
        static_topk = 40
        static_indices =high_similarity_indices[ :static_topk]
        # 将 cache_indices 转换为集合
        all_patches_to_highlight = set(all_patches_to_highlight)

        add = set(range(1, 257)) - set(high_similarity_indices)
        #  添加 high_similarity_indices 中不在 cache_indices 的元素
        all_patches_to_highlight |= add
        #  移除 cache_indices 中存在于 static_indices 的元素
        all_patches_to_highlight -= set(static_indices)

        all_patches_to_highlight = list(all_patches_to_highlight)

        cache_indices_group = [(all_patches_to_highlight, (180, 40, 50))]
        result_image3 = np.array(draw_patches_overlay(img, cache_indices_group, patch_size=14, alpha=0.4))
        dir3 = save_rollout_frame(result_image3, 35, frame_idx, total_episodes, log_file=None, view="primary")

        patches_sum_group = [(patches_sum, (15, 67, 100))]
        overlap_group = [(overlap, (200, 40, 50))]
        high_similarity_group = [(high_similarity_indices, (50, 196, 15))]
        result_image2 = draw_patches_overlay(img, overlap_group, patch_size=14, alpha=0.4)
        result_image2 = draw_patches_overlay(result_image2, patches_sum_group, patch_size=14, alpha=0.4)
        result_image2 = np.array(draw_patches_overlay(result_image2, high_similarity_group, patch_size=14, alpha=0.4))
        with open("./cache_indices.txt", "a") as f:
            f.write(f"all_patches_to_highlight:{patches_sum}\n")
        dir2 = save_rollout_frame(result_image2, 33, frame_idx, total_episodes, log_file=None, view="primary")
    return dir



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