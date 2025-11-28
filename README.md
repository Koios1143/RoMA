# RoMA (Routing Manifold Alignment)

Sparse Mixture-of-Experts (MoE) have been widely adopted in recent large language models since it can efficiently scale up the model capability without increasing the inference cost. However, evaluations on broad downstream tasks reveal a consistent suboptimality of the routers in existing MoE LLMs, which results in a severe performance gap (e.g., 10-20% in accuracy) to the optimal routing. In this paper, we show that aligning the manifold of routing weights with that of task embedding via post-training can effectively reduce the gap and improve MoE LLMs’ generalization performance. Our method, “Routing Manifold Alignment (RoMA)”, introduces an additional manifold regularization term in the post-training objective and only requires lightweight finetuning of routers (with other parameters frozen). Specifically, the regularization encourages the routing weights of each sample to be close to those of its successful neighbors (whose routing weights lead to correct answers) in a task embedding space. Consequently, samples targeting similar tasks will share similar expert choices across layers. Building such bindings between tasks and experts over different samples is essential to achieve better generalization. Moreover, RoMA demonstrates the advantage of unifying the task understanding (by embedding models) with solution generation (by MoE LLMs). In experiments, we finetune routers in two recent MoE LLMs using RoMA. Evaluations on diverse benchmarks and extensive comparisons with baselines show the substantial improvement brought by RoMA.

## 1) Environment

Compares to the original implementation, since we're using AMD MI300X GPU with ROCm 6.1, we use the following environment setup

- Python 3.11.14 (which match the original requirement 3.10+)
- PyTorch 2.4.1 (which match the original requirement 2.1+, but not CUDA)
- 
- Transformers (v4.57.3), Sentence-Transformers(v5.1.2)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.1
uv pip install transformers sentence-transformers datasets
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

I ran the finetune script with `bash finetune.sh`

Note that since I've faced some errors when running the code, described in [Finetune_error.md](./Finetune_error.md), I did some modification on `finetune_roma.py`.

After run the default setting, I get

```
Epoch 1: CE=2.2177, Reg=0.0044
Epoch 2: CE=1.8043, Reg=0.0044
Epoch 3: CE=1.4138, Reg=0.0044
Saved router checkpoint to ckpts/olmoe_roma_last5/router_ft.pt
```

**Multi-GPU (DDP)**

```bash
torchrun --nproc_per_node=4 finetune_roma.py <same args> --ddp
```

**Outputs**

- `ckpts/olmoe_roma_last5/router_ft.pt` — router-only checkpoint with updated gate weights for the last 5 layers.

------

## 4) Evaluation

```bash
python evaluate.py \
  --model allenai/OLMoE-1B-7B-0125-Instruct \
  --router_ckpt ckpts/olmoe_roma_last5/router_ft.pt \
  --batch_size 64
```

> Note that I removed the `--data data/test.jsonl` since there's no `--data` flag for `evaluate.py`

After running the evaluation, I have the following result:

```
ARC-Challenge (1150 ex) Accuracy: 22.87% (263/1150)
```

The report accuracy **22.87%** is way lower than expected.

I also did some modification on `evaluate.py`, such that we can evaluate the base model performance by not passing the `router_ckpt`. Run the following command for testing base model.

```bash
python evaluate.py \
  --model allenai/OLMoE-1B-7B-0125-Instruct \
  --batch_size 64
```

And obtain the following result:

```
ARC-Challenge (1150 ex) Accuracy: 22.87% (263/1150)
```

Which is identical to the finetuned one.

------
