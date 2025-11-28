python finetune_roma.py \
  --model allenai/OLMoE-1B-7B-0125-Instruct \
  --train_files data/instruct_openbookqa_correct_routing_results.json \
  --output_dir ckpts/olmoe_roma_last5 \
  --epochs 3 --batch_size 32 --lr 7e-5 --weight_decay 0.01 \
  --lambda_reg 0.8 --k 5 --sigma 0.4 \
  --warmup_steps 2000 --lambda_warmup_steps 2000 --max_len 1024