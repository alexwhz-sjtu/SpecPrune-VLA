
'''
/share/public/wanghanzhen/anaconda3/envs/openvla-oft-o/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py

new_indices_primary = self.high_attn_score(layer_attention, topk=ATTN_TOPK_PRECISE if precise_mode else ATTN_TOPK, primary=True) # -> set()
new_indices_wrist = self.high_attn_score(layer_attention, topk=ATTN_TOPK_PRECISE if precise_mode else ATTN_TOPK_WRIST, primary=False)

'''

ATTN_TOPK = 20
ATTN_TOPK_WRIST = 20
ATTN_TOPK_PRECISE = 40

PRIMARY_TOPK = 220
PRIMARY_TOPK_PRECISE = 200
WRIST_TOPK = 220
WRIST_TOPK_PRECISE = 220

'''
/share/public/wanghanzhen/openvla-oft-o/experiments/robot/openvla_utils.py

primary_topk = PRIMARY_TOPK_PRECISE if precise_mode else PRIMARY_TOPK
wrist_topk = WRIST_TOPK_PRECISE if precise_mode else WRIST_TOPK

high_similarity_indices = get_similarity_indices(result_image[0], prev_images[0], top_k=primary_topk, patch_size=14, sim_threshold=0.990, view="primary")
high_similarity_indices_wrist = get_similarity_indices(result_image[1], prev_images[1], top_k=wrist_topk, patch_size=14, sim_threshold=0.992, view="wrist")

'''