MODEL="allenai/OLMoE-1B-7B-0125-Instruct"
CKPT="./ckpts/olmoe_roma_last5/router_ft.pt"
MODEL_ARGS="pretrained=$MODEL,trust_remote_code=True,checkpoint_path=$CKPT"

python custom_lm_eval.py \
    --model roma \
    --model_args $MODEL_ARGS \
    --tasks arc_challenge \
    --device cuda:0 \
    --batch_size 8 \
    --seed 42