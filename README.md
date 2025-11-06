# RoMA (Routing Manifold Alignment)

------

## 1) Environment

- Python 3.10+
- PyTorch 2.1+ with CUDA
- Transformers, Sentence-Transformers

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers sentence-transformers
```

> Model tested: `allenai/OLMoE-1B-7B-0125-Instruct`

------

## 2) Data format

**Training inputs** (SciQ/OpenBookQA) use a **JSON dict** of routing results (NOT JSONL):

```json
{
  "7-870": {
    "input_text": "<prompt with question+choices>",
    "last_token_routing": {
      "token_id": 27,
      "token_text": ":",
      "layers": [
        { "layer": 11, "routing_weights": {"expert_0": 0.02, "expert_1": 0.03, "expert_2": 0.05} },
        { "layer": 12, "routing_weights": {"expert_0": 0.01, "expert_1": 0.07} }
      ]
    },
    "correct_answer": "A",
    "model_answer": "B",
    "is_correct": true
  }
}
```

- `is_correct == true` entries form the **successful set S**.

**Evaluation input**  uses **JSONL** with fields:

```json
{"question": "...", "choices": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "A"}
```

------

## 3) Finetune (RoMA)

By default we update only the **last 5 layers** routers (`mlp.gate`) and optimize:

$\mathcal{L}_{\text{RoMA}} = \mathcal{L}_{\text{task}} + \lambda(t) \cdot \sum_{j \in \mathcal{N}(i)} W_{ij} \lVert r_i - r_j \rVert_2^2$

```bash
python finetune_roma.py \
  --model allenai/OLMoE-1B-7B-0125-Instruct \
  --train_files data/instruct_openbookqa_correct_routing_results.json \
  --output_dir ckpts/olmoe_roma_last5 \
  --epochs 3 --batch_size 32 --lr 7e-5 --weight_decay 0.01 \
  --lambda_reg 0.8 --k 5 --sigma 0.4 \
  --warmup_steps 2000 --lambda_warmup_steps 2000 --max_len 1024
```

**Multi-GPU (DDP)**

```bash
torchrun --nproc_per_node=4 finetune_roma.py <same args> --ddp
```

**Outputs**

- `ckpts/olmoe_roma_last5/router_ft.pt` — router-only checkpoint with updated gate weights for the last 5 layers.

------

## 4) Evaluate on ARC-Challenge

```bash
python evaluate.py \
  --model allenai/OLMoE-1B-7B-0125-Instruct \
  --router_ckpt ckpts/olmoe_roma_last5/router_ft.pt \
  --data data/test.jsonl --batch_size 64
```



------
