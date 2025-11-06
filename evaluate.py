import argparse
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, OlmoeForCausalLM

from datasets import load_dataset

CHOICE_LETTERS = ["A","B","C","D"]


def build_prompt(q: str, choices: List[str]) -> str:
    lines = [
        "Answer with only a single letter (A, B, C, or D).",
        "Question:", q,
        "Choices:",
    ]
    for i, c in enumerate(choices):
        lines.append(f"{CHOICE_LETTERS[i]}) {c}")
    lines.append("Answer:")
    return "\n".join(lines)


def extract_answer_letter(text: str) -> str:
    text = (text or "").strip().upper()
    for ch in CHOICE_LETTERS:
        if text.startswith(ch):
            return ch
    for tok in text.replace(".", ")").split():
        if tok and tok[0] in CHOICE_LETTERS:
            return tok[0]
    return ""


def normalize_hf_example(ex) -> Tuple[str, List[str], str]:
    """Return (question, [choices...], answer_letter). Robust to schema variants."""
    q = (ex.get("question") or "").strip()

    raw_choices = ex.get("choices")
    texts = []
    if isinstance(raw_choices, dict) and "text" in raw_choices:
        texts = [str(t) for t in raw_choices["text"]]
    elif isinstance(raw_choices, list):
        for c in raw_choices:
            if isinstance(c, dict) and "text" in c:
                texts.append(str(c["text"]))
            else:
                texts.append(str(c))
    else:
        # fallback
        if raw_choices is not None:
            texts = [str(raw_choices)]
        else:
            texts = []

    texts = texts[:4]
    if len(texts) < 4:
        texts = texts + [""] * (4 - len(texts))

    # answer key
    ans = (ex.get("answerKey") or ex.get("answer") or "").strip().upper()[:1]
    if ans not in CHOICE_LETTERS:
        ans = ""

    return q, texts, ans


class ARCDatasetHF(Dataset):
    def __init__(self, split_name: str = "test", max_examples: int = None):
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split_name)
        self.items = []
        for ex in ds:
            q, choices, ans = normalize_hf_example(ex)
            if q and choices and ans:
                self.items.append((q, choices, ans))
            if max_examples is not None and len(self.items) >= max_examples:
                break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


@torch.no_grad()
def load_with_router_ckpt(model_name: str, ckpt_path: str, device: str):
    model = OlmoeForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)
    # load gates
    payload = torch.load(ckpt_path, map_location='cpu')
    target_layers = payload['target_layers']
    state = payload['state_dict']
    for li in target_layers:
        gate = model.model.layers[li].mlp.gate
        w_key = f'layers.{li}.mlp.gate.weight'
        b_key = f'layers.{li}.mlp.gate.bias'
        gate.weight.data.copy_(state[w_key])
        if gate.bias is not None and b_key in state:
            gate.bias.data.copy_(state[b_key])
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, tokenizer, dataset: ARCDatasetHF, batch_size: int, device: str, max_new_tokens: int = 1):
    def collate(batch):
        prompts = [build_prompt(q, c) for q, c, _ in batch]
        labels = [CHOICE_LETTERS.index(a) for _, _, a in batch]
        enc = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        return enc['input_ids'], enc['attention_mask'], torch.tensor(labels)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    correct = 0
    total = 0
    for input_ids, attn_mask, labels in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        preds = [extract_answer_letter(t) for t in dec]
        for p, l in zip(preds, labels):
            total += 1
            if p == CHOICE_LETTERS[l.item()]:
                correct += 1
    acc = correct / max(1, total)
    print(f"ARC-Challenge ({len(dataset)} ex) Accuracy: {acc*100:.2f}% ({correct}/{total})")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='allenai/OLMoE-1B-7B-0125-Instruct')
    parser.add_argument('--router_ckpt', type=str, required=True)
    parser.add_argument('--hf_split', type=str, default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_examples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_new_tokens', type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_with_router_ckpt(args.model, args.router_ckpt, args.device)

    ds = ARCDatasetHF(split_name=args.hf_split, max_examples=args.max_examples)

    evaluate(model, tokenizer, ds, args.batch_size, args.device, max_new_tokens=args.max_new_tokens)


if __name__ == '__main__':
    main()
