import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_DIR = Path(os.path.expanduser("~/cephalon-data/models"))
EMBEDDER_ID = "jinaai/jina-embeddings-v5-text-small"
RERANKER_ID = "jinaai/jina-reranker-v3"
EMBEDDER_SEQUENCE_LENGTH = 512


class JinaRetrievalEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids)
        hidden = outputs.last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), sequence_lengths]
        return F.normalize(pooled, p=2, dim=-1)


def replace_dir(temp_dir: Path, final_dir: Path) -> None:
    backup_dir = final_dir.with_name(f"{final_dir.name}.backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if final_dir.exists():
        final_dir.rename(backup_dir)
    temp_dir.rename(final_dir)


def export_embedder(model_dir: Path, force: bool) -> None:
    final_dir = model_dir / "embedder"
    if final_dir.joinpath("model.onnx").exists() and not force:
        print(f"Embedder already exists at {final_dir}")
        return

    temp_dir = model_dir / "embedder.exporting"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting embedder with custom Jina retrieval wrapper: {EMBEDDER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDER_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(EMBEDDER_ID, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager")
    if hasattr(model, "set_adapter"):
        model.set_adapter(["retrieval"])
    model.eval()

    wrapper = JinaRetrievalEmbeddingWrapper(model).eval()
    dummy = tokenizer(
        ["Document: Cephalon export validation text."],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=EMBEDDER_SEQUENCE_LENGTH,
    )
    input_names = ["input_ids", "attention_mask"]
    output_names = ["embedding"]
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(temp_dir / "model.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "embedding": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
        external_data=True,
        dynamo=False,
    )
    tokenizer.save_pretrained(temp_dir)
    temp_dir.joinpath("cephalon_onnx_meta.json").write_text(
        json.dumps({
            "model_id": EMBEDDER_ID,
            "kind": "embedder",
            "pooling": "last_token",
            "normalized": True,
            "dimension": 1024,
            "fixed_sequence_length": EMBEDDER_SEQUENCE_LENGTH,
        }, indent=2),
        encoding="utf-8",
    )
    replace_dir(temp_dir, final_dir)


def export_reranker(model_dir: Path, force: bool) -> None:
    final_dir = model_dir / "reranker"
    if final_dir.joinpath("model.onnx").exists() and not force:
        print(f"Reranker already exists at {final_dir}")
        return

    temp_dir = model_dir / "reranker.exporting"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting reranker: {RERANKER_ID}")
    reranker_model = ORTModelForSequenceClassification.from_pretrained(RERANKER_ID, export=True, trust_remote_code=True)
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_ID, trust_remote_code=True)
    reranker_model.save_pretrained(temp_dir)
    reranker_tokenizer.save_pretrained(temp_dir)
    temp_dir.joinpath("cephalon_onnx_meta.json").write_text(
        json.dumps({"model_id": RERANKER_ID, "kind": "reranker"}, indent=2),
        encoding="utf-8",
    )
    replace_dir(temp_dir, final_dir)


def export_model(model_dir: Path, force: bool = False, skip_embedder: bool = False, skip_reranker: bool = False) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    if not skip_embedder:
        export_embedder(model_dir, force)
    if not skip_reranker:
        export_reranker(model_dir, force)
    print("ONNX export complete.")
    print("Model license note: Jina models are CC BY-NC 4.0; keep notices with packaged builds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Cephalon ONNX embedder and reranker models.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Target model directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing ONNX exports.")
    parser.add_argument("--skip-embedder", action="store_true", help="Skip embedder export.")
    parser.add_argument("--skip-reranker", action="store_true", help="Skip reranker export.")
    args = parser.parse_args()
    export_model(
        Path(args.model_dir).expanduser().resolve(),
        force=args.force,
        skip_embedder=args.skip_embedder,
        skip_reranker=args.skip_reranker,
    )
