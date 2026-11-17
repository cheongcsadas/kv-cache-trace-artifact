"""Load the pinned upstream vector cache and decoding routine without package side effects."""
import ast
import hashlib
import math
import os
from pathlib import Path
import threading
from types import SimpleNamespace

REVISION = "29ef1974fcb5a1440ccf92d7d7996900230fddcf"
SOURCE_PATH = "minference/modules/retr_attn.py"
SOURCE_SHA256 = "c8980c02232f8a52b23152804a5f3ab509683116b742c03b86c998d2af81109b"


def source_text(source_dir):
    path = Path(source_dir) / SOURCE_PATH
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    if hashlib.sha256(text.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError(f"RetrievalAttention source does not match pinned commit {REVISION}")
    return text


def load_native(source_dir, flash_kernel=None):
    import torch
    from .index import FlatIndex
    if flash_kernel is None:
        from flash_attn import flash_attn_func
        flash_kernel = flash_attn_func
    text = source_text(source_dir)
    # Fix two numerical issues in the upstream merge; preserve its CPU retrieval,
    # partial attention, GPU attention, and log-normalizer-weighted combination.
    replacements = {
        "q = q.detach().cpu().numpy()": "q = q.float().detach().cpu().numpy()",
        "retrieval_lse = torch.log(torch.sum(torch.exp(retrieval_attn_weights), dim=-1)).to(":
            "retrieval_lse = torch.logsumexp(retrieval_attn_weights, dim=-1).to(",
        "retrieval_out[:, head_idx:, q_idx, :]": "retrieval_out[:, head_idx, q_idx, :]",
        "retrieval_lse + torch.log(1 + torch.exp(flash_lse - retrieval_lse))":
            "torch.logaddexp(retrieval_lse, flash_lse)",
        "    core = os.cpu_count() // 2\n\n    device = q.device":
            "    core = decoding_kwargs.get('cpu_threads', max(1, (os.cpu_count() or 2)//2))\n\n    device = q.device",
    }
    for before, after in replacements.items():
        if text.count(before) != 1:
            raise ValueError("Pinned upstream numerical patch no longer matches")
        text = text.replace(before, after)
    parsed = ast.parse(text, filename=SOURCE_PATH)
    nodes = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in {"VectorDB_KV_Cache", "retr_attn"}]
    if len(nodes) != 2:
        raise ValueError("Required native RetrievalAttention definitions are absent")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = dict(torch=torch, nn=torch.nn, math=math, os=os, threading=threading,
                     papyfaiss=SimpleNamespace(FlatIndex=FlatIndex), flash_attn_func=flash_kernel)
    exec(compile(module, str(Path(source_dir) / SOURCE_PATH), "exec"), namespace)
    return SimpleNamespace(VectorDB_KV_Cache=namespace["VectorDB_KV_Cache"],
                           retr_attn=namespace["retr_attn"])
