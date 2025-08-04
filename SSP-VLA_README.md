## to SETUP

1. follow the [README.md](openvla-oft/README.md) and [SETUP.md](openvla-oft/SETUP.md) to create conda environment and dependencies
2. follow the[ LIBERO.md](openvla-oft/LIBERO.md) to setup simulator and enable inference in LIBERO simulative environment( in dir openvla-oft)
3. substitute the file modeling_llama.py in ./anaconda3/envs/openvla-oft-o/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py with[ modeling_llama.py](modeling_llama.py) in this repo

to lauch inference task in LIBERO:

the checkpoint path has been modified to the local path

```
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /home/wanghanzhen/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-10/snapshots/95220f9a3421a7ff12d4218e73d09ade830fa9a3 \
  --task_suite_name libero_10

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /home/wanghanzhen/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-spatial/snapshots/6d0231af0e48c5985f1ff86908f4674b84bc049b \
  --task_suite_name libero_spatial

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /home/wanghanzhen/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b \
  --task_suite_name libero_goal

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /home/wanghanzhen/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-object/snapshots/4c89574e1c538b6c102f43f0526d60a9d3650148 \
  --task_suite_name libero_object
```
