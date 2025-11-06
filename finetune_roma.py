import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, OlmoeForCausalLM
from sentence_transformers import SentenceTransformer, util

CHOICE_LETTERS = ["A","B","C","D"]

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0

@dataclass
class TrainItem:
    input_text: str
    correct_answer: str  
    is_correct: bool
    routing_layers: List[Dict[str, Any]] 

class RoutingJSONDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.items: List[TrainItem] = []
        for p in paths:
            with open(p, 'r', encoding='utf-8') as f:
                blob = json.load(f)
            for _id, ex in blob.items():
                input_text = (ex.get("input_text") or "").strip()
                correct_answer = (ex.get("correct_answer") or "").strip().upper()
                is_correct = bool(ex.get("is_correct", False))
                ltr = ex.get("last_token_routing", {})
                layers = ltr.get("layers", [])
                self.items.append(TrainItem(input_text, correct_answer, is_correct, layers))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def extract_answer_letter(text: str) -> str:
    text = (text or "").strip().upper()
    for ch in CHOICE_LETTERS:
        if text.startswith(ch):
            return ch
    for tok in text.replace(".", ")").split():
        if tok and tok[0] in CHOICE_LETTERS:
            return tok[0]
    return ""

def r_from_layers_json(layers: List[Dict[str, Any]], num_experts: int, target_layers: List[int], device: torch.device):
    # Map layer_id -> probs tensor [E]
    probs_by_layer: Dict[int, torch.Tensor] = {}
    for li in target_layers:
        probs_by_layer[li] = None
    for L in layers:
        li = int(L.get("layer", -1))
        if li in probs_by_layer:
            rw = L.get("routing_weights", {})
            # Build vector length num_experts, fill 0, then assign expert_i
            vec = torch.zeros(num_experts, dtype=torch.float32)
            for k, v in rw.items():
                if not k.startswith("expert_"):
                    continue
                try:
                    ei = int(k.split("_")[-1])
                    if 0 <= ei < num_experts:
                        vec[ei] = float(v)
                except Exception:
                    pass
            # normalize to sum=1
            s = vec.sum()
            if s > 0:
                vec = vec / s
            probs_by_layer[li] = vec
    filled = []
    for li in target_layers:
        v = probs_by_layer[li]
        if v is None:
            v = torch.full((num_experts,), 1.0/num_experts, dtype=torch.float32)
        filled.append(v)
    return torch.cat(filled, dim=-1).to(device)  # [E*L]

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='allenai/OLMoE-1B-7B-0125-Instruct')
    parser.add_argument('--train_files', type=str, nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lambda_reg', type=float, default=0.8)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=0.4)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--lambda_warmup_steps', type=int, default=2000)
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # DDP init
    if args.ddp:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = OlmoeForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.to(device)
    model.train()

    # choose LAST 5 layers
    num_layers = len(model.model.layers)
    target_layers = list(range(max(0, num_layers - 5), num_layers))
    num_experts = model.model.layers[0].mlp.num_experts

    # Freeze all, unfreeze only gates of target layers
    for p in model.parameters():
        p.requires_grad = False
    router_params = []
    for li in target_layers:
        for p in model.model.layers[li].mlp.gate.parameters():
            p.requires_grad = True
            router_params.append(p)

    optimizer = torch.optim.AdamW(router_params, lr=args.lr, weight_decay=args.weight_decay)

    # LR warmup schedule
    def lr_lambda(step):
        if args.warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / float(args.warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # lambda warmup schedule
    def lambda_factor(step):
        if args.lambda_warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / float(args.lambda_warmup_steps))

    # Losses
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')  # per-sample

    # Data
    ds = RoutingJSONDataset(args.train_files)
    if args.ddp:
        sampler = DistributedSampler(ds, shuffle=True)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Build successful set S from training JSONs directly (is_correct==True)
    if is_main_process():
        print("Building successful set S from provided routing JSONs...")
    S_texts: List[str] = []
    S_r_list: List[torch.Tensor] = []
    for it in ds.items:
        if it.is_correct:
            S_texts.append(it.input_text)
            S_r_list.append(r_from_layers_json(it.routing_layers, num_experts, target_layers, device=torch.device('cpu')))
    if len(S_texts) == 0 and is_main_process():
        print("Warning: Successful set S is empty; manifold loss will be zero.")

    # Precompute embeddings and r_j for S
    if len(S_texts) > 0:
        embedder = get_embedder()
        with torch.no_grad():
            S_emb = embedder.encode(S_texts, convert_to_tensor=True, device=device)
        S_r = torch.stack(S_r_list, dim=0).to(device)  # [|S|, E*L]

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Training
    global_step = 0
    for epoch in range(args.epochs):
        if args.ddp:
            loader.sampler.set_epoch(epoch)
        running_ce, running_reg, running_cnt = 0.0, 0.0, 0

        for batch in loader:
            input_texts: List[str] = batch.input_text
            golds: List[str] = batch.correct_answer

            # Encode inputs (truncate) if available for CE
            enc = None
            labels_idx = None
            usable_mask = []
            if any(t for t in input_texts) and any(extract_answer_letter(a) for a in golds):
                prompts = []
                labels_idx = []
                for txt, ans in zip(input_texts, golds):
                    if txt and extract_answer_letter(ans):
                        prompts.append(txt)
                        labels_idx.append(CHOICE_LETTERS.index(extract_answer_letter(ans)))
                        usable_mask.append(True)
                    else:
                        prompts.append("")
                        labels_idx.append(0)
                        usable_mask.append(False)
                enc = tokenizer(prompts, padding=True, truncation=True, max_length=args.max_len, return_tensors='pt')
                enc = {k: v.to(device) for k, v in enc.items()}
                labels_idx_t = torch.tensor(labels_idx, device=device)
            else:
                usable_mask = [False] * len(input_texts)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                ce = torch.tensor(0.0, device=device)
                if enc is not None:
                    outputs = model(**enc)
                    logits = outputs.logits[:, -1, :]  # [B, V]
                    # project to letters via next-token logits of ' A',' B',' C',' D'
                    cand_ids = [tokenizer.encode(" " + ch, add_special_tokens=False)[0] for ch in CHOICE_LETTERS]
                    cand_logits = logits.index_select(-1, torch.tensor(cand_ids, device=device))  # [B,4]
                    per_sample_ce = ce_loss_fn(cand_logits, labels_idx_t)  # [B]
                    mask_t = torch.tensor([1 if u else 0 for u in usable_mask], device=device, dtype=per_sample_ce.dtype)
                    if mask_t.sum() > 0:
                        ce = (per_sample_ce * mask_t).sum() / mask_t.sum()

                reg = torch.tensor(0.0, device=device)
                if len(S_texts) > 0:
                    # r_i for current batch from JSON (NOT model hooks; faithful to your files)
                    r_i_list = [r_from_layers_json(it.routing_layers, num_experts, target_layers, device=device) for it in batch]
                    r_i = torch.stack(r_i_list, dim=0)  # [B, E*L]
                    # neighbor weights W_{i,j}
                    embedder = get_embedder()
                    with torch.no_grad():
                        Q_emb = embedder.encode(input_texts, convert_to_tensor=True, device=device)
                        scores = util.cos_sim(Q_emb, S_emb)  # [B, |S|]
                        k_eff = min(args.k, scores.shape[1])
                        topk = torch.topk(scores, k=k_eff, dim=1)
                        idx = topk.indices  # [B, k]
                        sim = topk.values    # [B, k]
                        # Gaussian similarity over cosine (turn into [0,1] distance proxy)
                        if args.sigma > 0:
                            sim = torch.exp(-(1 - sim) ** 2 / (2 * (args.sigma ** 2)))
                        W = sim / (sim.sum(dim=1, keepdim=True) + 1e-9)
                    r_neighbors = S_r.index_select(0, idx.flatten()).view(r_i.size(0), -1, r_i.size(1))  # [B,k,E*L]
                    diff = r_i.unsqueeze(1) - r_neighbors  # [B,k,E*L]
                    reg = (W.unsqueeze(-1) * (diff * diff)).sum() / r_i.size(0)

                lam = args.lambda_reg * lambda_factor(global_step)
                loss = ce + lam * reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            # logging
            if is_main_process():
                running_ce += ce.detach().item() * len(input_texts)
                running_reg += reg.detach().item() * len(input_texts)
                running_cnt += len(input_texts)

        if is_main_process() and running_cnt > 0:
            print(f"Epoch {epoch+1}: CE={running_ce/running_cnt:.4f}, Reg={running_reg/running_cnt:.4f}")

    # Save checkpoint
    if is_main_process():
        save_path = os.path.join(args.output_dir, 'router_ft.pt')
        state = {}
        for li in target_layers:
            gate = model.model.layers[li].mlp.gate
            state[f'layers.{li}.mlp.gate.weight'] = gate.weight.detach().cpu()
            if gate.bias is not None:
                state[f'layers.{li}.mlp.gate.bias'] = gate.bias.detach().cpu()
        torch.save({'target_layers': target_layers, 'state_dict': state, 'model_name': args.model}, save_path)
        print(f"Saved router checkpoint to {save_path}")

    # DDP cleanup
    if args.ddp:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
