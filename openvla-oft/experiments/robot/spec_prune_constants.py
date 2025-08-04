
'''
/share/public/wanghanzhen/anaconda3/envs/openvla-oft-o/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py

new_indices_primary = self.high_attn_score(layer_attention, topk=ATTN_TOPK_PRECISE if precise_mode else ATTN_TOPK, primary=True) # -> set()
new_indices_wrist = self.high_attn_score(layer_attention, topk=ATTN_TOPK_PRECISE if precise_mode else ATTN_TOPK_WRIST, primary=False)

'''
DYNAMIC_PRUNE_RATIO = 0.8
STATIC_PRUNE_RATIO = 0.5

# ATTN_TOPK = 20
# ATTN_TOPK_WRIST = 20
# ATTN_TOPK_PRECISE = 30
# 15,15,25

ATTN_TOPK = int(20*STATIC_PRUNE_RATIO)
ATTN_TOPK_WRIST = int(15*STATIC_PRUNE_RATIO)
if STATIC_PRUNE_RATIO <= 1.0:
    ATTN_TOPK_PRECISE = int(40*STATIC_PRUNE_RATIO**2)
    ATTN_TOPK_PRECISE_WRIST = int(30*STATIC_PRUNE_RATIO**2)
else:
    ATTN_TOPK_PRECISE = int(40*STATIC_PRUNE_RATIO)
    ATTN_TOPK_PRECISE_WRIST = int(30*STATIC_PRUNE_RATIO)

# 220 200 220 220
PRIMARY_TOPK = 230
PRIMARY_TOPK_PRECISE = 200
WRIST_TOPK = 230
WRIST_TOPK_PRECISE = 220


VELOCITY_THRES = 0.4
ROT_THRES = 0.2
XY_THRES = 0.1
Z_THRES = 0.1

SKIP_LAYER=0
'''
/share/public/wanghanzhen/openvla-oft-o/experiments/robot/openvla_utils.py

primary_topk = PRIMARY_TOPK_PRECISE if precise_mode else PRIMARY_TOPK
wrist_topk = WRIST_TOPK_PRECISE if precise_mode else WRIST_TOPK

high_similarity_indices = get_similarity_indices(result_image[0], prev_images[0], top_k=primary_topk, patch_size=14, sim_threshold=0.990, view="primary")
high_similarity_indices_wrist = get_similarity_indices(result_image[1], prev_images[1], top_k=wrist_topk, patch_size=14, sim_threshold=0.992, view="wrist")

'''