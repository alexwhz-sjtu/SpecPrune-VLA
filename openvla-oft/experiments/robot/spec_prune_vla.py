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

    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width = image.size[0]
    num_patches = width // patch_size

    for patch_list, color in patch_groups:

        border_color = (
            min(int(color[0] * 1.2), 255), 
            min(int(color[1] * 1.2), 255),
            min(int(color[2] * 1.2), 255),
            255
        )
        
        for pid in patch_list:
            i, j = divmod(pid, num_patches)
            top_left = (j * patch_size, i * patch_size)
            bottom_right = ((j + 1) * patch_size, (i + 1) * patch_size)

            draw.rectangle([top_left, bottom_right], 
                         fill=color + (int(255 * alpha),))
            

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
    
    attn_dict = {}
    token_position = {}
    num_tokens = 0
    for layer_idx in layer_indices:
        attn_dict[layer_idx] = {}
        attn_map = multihead_attention[layer_idx][0].to(torch.float32).squeeze(0).mean(dim=0)
        token_idx = multihead_attention[layer_idx][1]

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
    total_patches = 256 + offset

    for layer_idx in attn_dict:
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
        

    return torch.tensor(top_attn_patches) + offset


