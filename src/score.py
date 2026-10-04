"""Score every prompt in sets A and B: log10 P(reply leaks the secret), by teacher forcing.

Leak = the assistant reply begins with one of `behavior.leak_prefixes` in config.yaml. The prefixes
are mutually exclusive, so the summed probability is a lower bound on P(leak). The first prefix
alone (the bare secret) is the "probability of a specific output" proxy of Jones et al.,
"Forecasting Rare Language Model Behaviors", arXiv:2502.16797, Section 4.1.

Resumable: scores are written in chunks to results/scores/<set>/part_*.parquet, and finished
chunks are skipped on restart. When a set is complete, results/scores_<set>.parquet is written.

Usage:  python src/score.py          or, in Jupyter:  from src.score import run; run()
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 2000


class Scorer:
    def __init__(self, cfg: dict):
        b, m = cfg["behavior"], cfg["model"]
        self.system = b["system_prompt"]
        self.prefixes = b["leak_prefixes"]
        assert not any(x != y and y.startswith(x) for x in self.prefixes for y in self.prefixes), \
            "leak prefixes must be mutually exclusive"
        rev = m.get("revision")
        rev = None if (not rev or str(rev).startswith("TODO")) else rev
        self.tok = AutoTokenizer.from_pretrained(m["name"], revision=rev, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            m["name"], revision=rev, dtype=torch.bfloat16, device_map="cuda").eval()
        self.commit = getattr(self.model.config, "_commit_hash", None) or rev
        self.prefix_ids = [self.tok(p, add_special_tokens=False)["input_ids"] for p in self.prefixes]

    def chat_ids(self, text: str) -> list[int]:
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": text}]
        s = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return self.tok(s, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def score_target(self, texts: list[str], target: list[int], batch_size: int = 32) -> np.ndarray:
        """Natural-log P(reply starts with `target` | system prompt, user text)."""
        S, pad, out = len(target), self.tok.pad_token_id, []
        for i in range(0, len(texts), batch_size):
            seqs = [self.chat_ids(t) + target for t in texts[i:i + batch_size]]
            L = max(len(s) for s in seqs)
            ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device="cuda")
            mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device="cuda")
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            logits = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                logits_to_keep=S + 1).logits
            logp = torch.log_softmax(logits[:, -S - 1:-1, :].float(), dim=-1)
            out.extend(logp.gather(-1, ids[:, -S:].unsqueeze(-1)).squeeze(-1).sum(-1).tolist())
        return np.array(out)

    def score(self, texts: list[str], batch_size: int = 32) -> pd.DataFrame:
        per = np.stack([self.score_target(texts, t, batch_size) for t in self.prefix_ids], axis=1)
        df = pd.DataFrame(per / np.log(10), columns=[f"lp_{k}" for k in range(len(self.prefixes))])
        df["log10p_any"] = np.logaddexp.reduce(per, axis=1) / np.log(10)
        df["log10p_start"] = df["lp_0"]
        return df


def score_set(scorer: Scorer, name: str, prompts: pd.DataFrame) -> pd.DataFrame:
    part_dir = ROOT / "results" / "scores" / name
    part_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = -(-len(prompts) // CHUNK)
    t0 = time.time()
    for c in range(n_chunks):
        path = part_dir / f"part_{c:05d}.parquet"
        if path.exists():
            continue
        chunk = prompts.iloc[c * CHUNK:(c + 1) * CHUNK].reset_index(drop=True)
        s = scorer.score(chunk.text.tolist())
        s.insert(0, "prompt_id", chunk.prompt_id)
        s.to_parquet(path, index=False)
        print(f"  set {name}: chunk {c + 1}/{n_chunks} done ({time.time() - t0:.0f} s this session)", flush=True)
    parts = sorted(part_dir.glob("part_*.parquet"))
    scores = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    assert len(scores) == len(prompts) and scores.prompt_id.is_unique, "incomplete or duplicated chunks"
    meta = prompts.drop(columns=["text"])
    out = meta.merge(scores, on="prompt_id", validate="one_to_one")
    out.to_parquet(ROOT / "results" / f"scores_{name}.parquet", index=False)
    return out


def run(sets: tuple[str, ...] = ("A", "B")) -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    scorer = Scorer(cfg)
    prompt_sets = {n: pd.read_parquet(ROOT / "data" / f"set_{n.lower()}.parquet") for n in sets}
    info = dict(model=cfg["model"]["name"], commit=scorer.commit, system_prompt=scorer.system,
                leak_prefixes=scorer.prefixes,
                prompt_hash={n: hashlib.sha256("\n".join(df.text).encode()).hexdigest()[:16]
                             for n, df in prompt_sets.items()},
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    (ROOT / "results").mkdir(exist_ok=True)
    info_path = ROOT / "results" / "run_info.json"
    if info_path.exists():
        old = json.loads(info_path.read_text())
        for k in ("model", "commit", "system_prompt", "leak_prefixes", "prompt_hash"):
            if old.get(k) != info[k]:
                raise RuntimeError(f"config changed since earlier chunks were scored ({k}); "
                                   "delete results/scores/ and results/run_info.json to start over")
    info_path.write_text(json.dumps(info, indent=2))
    print(f"model commit {scorer.commit}")
    for name in sets:
        prompts = prompt_sets[name]
        print(f"scoring set {name}: {len(prompts):,} prompts")
        out = score_set(scorer, name, prompts)
        print(f"  set {name} done: median log10p_any {out.log10p_any.median():.2f}, "
              f"max {out.log10p_any.max():.2f}")


if __name__ == "__main__":
    run()
