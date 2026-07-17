import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def validate_embedder(model_dir: Path) -> dict:
    meta_path = find_meta_path(model_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    session = ort.InferenceSession(str(model_dir / "model.onnx"))
    fixed_length = meta.get("fixed_sequence_length")
    tokenizer_kwargs = {"truncation": True, "return_tensors": "np"}
    if fixed_length:
        tokenizer_kwargs.update({"padding": "max_length", "max_length": int(fixed_length)})
    else:
        tokenizer_kwargs["padding"] = True
    inputs = tokenizer(["Document: validation text"], **tokenizer_kwargs)
    input_names = {inp.name for inp in session.get_inputs()}
    ort_inputs = {"input_ids": inputs["input_ids"].astype(np.int64)}
    if "attention_mask" in input_names:
        ort_inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)
    if "token_type_ids" in inputs:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
    output = np.asarray(session.run(None, ort_inputs)[0])
    dim = int(output.shape[-1])
    norm = float(np.linalg.norm(output.reshape(-1, dim)[0]))
    return {"kind": "embedder", "path": str(model_dir), "output_shape": list(output.shape), "dimension": dim, "norm": round(norm, 6)}


def validate_reranker(model_dir: Path, max_batch_size: int = 1) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    session = ort.InferenceSession(str(model_dir / "model.onnx"))
    positive = ["breathing exercise", "The 4-7-8 breathing method is a breathing exercise."]
    negative = ["breathing exercise", "A graphics card renders pixels for video output."]
    input_names = {inp.name for inp in session.get_inputs()}

    def run_pairs(pairs: list[list[str]]) -> np.ndarray:
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="np")
        ort_inputs = {"input_ids": inputs["input_ids"].astype(np.int64)}
        if "attention_mask" in input_names:
            ort_inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)
        if "token_type_ids" in inputs and "token_type_ids" in input_names:
            ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
        return np.asarray(session.run(None, ort_inputs)[0])

    # Run calibration pairs individually because a valid export may have a
    # fixed singleton reshape. This determines positive-class orientation.
    raw_scores = np.concatenate([run_pairs([positive]), run_pairs([negative])])
    scores, score_mode = reranker_scores(raw_scores)
    if not scores or scores[0] <= scores[1]:
        raise RuntimeError("Reranker validation failed: the positive pair did not outrank the negative pair.")

    if max_batch_size > 1:
        batch_scores = run_pairs([positive] * max_batch_size)
        if batch_scores.shape[0] != max_batch_size:
            raise RuntimeError(f"Reranker batch validation returned {batch_scores.shape[0]} scores for {max_batch_size} pairs.")
    return {
        "kind": "reranker",
        "path": str(model_dir),
        "scores": [round(score, 6) for score in scores],
        "score_mode": score_mode,
        "max_batch_size": max_batch_size,
    }


def reranker_scores(raw_scores: np.ndarray) -> tuple[list[float], str]:
    if raw_scores.ndim == 2 and raw_scores.shape[1] == 2:
        candidates = [
            ("logit_margin_0_minus_1", raw_scores[:, 0] - raw_scores[:, 1]),
            ("logit_margin_1_minus_0", raw_scores[:, 1] - raw_scores[:, 0]),
            ("class_0", raw_scores[:, 0]),
            ("class_1", raw_scores[:, 1]),
        ]
        for mode, values in candidates:
            values = values.astype(float).tolist()
            if values[0] > values[1]:
                return values, mode
        return candidates[0][1].astype(float).tolist(), candidates[0][0]
    if raw_scores.ndim == 2 and raw_scores.shape[1] > 1:
        return raw_scores[:, -1].astype(float).tolist(), "class_last"
    return raw_scores.reshape(-1).astype(float).tolist(), "scalar"


def mark_validated(model_dir: Path, key: str, valid: bool, extra: dict | None = None) -> None:
    meta_path = find_meta_path(model_dir)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("kind", model_dir.name)
    meta.setdefault("model_id", model_dir.name)
    meta["validated"] = bool(valid)
    meta["validation_key"] = key
    if extra:
        meta.update(extra)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def model_hash(model_dir: Path) -> str:
    digest = hashlib.sha256()
    with (model_dir / "model.onnx").open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def find_meta_path(model_dir: Path) -> Path:
    generic = model_dir / "onnx_profile.json"
    if generic.exists():
        return generic
    legacy = model_dir / "cephalon_onnx_meta.json"
    if legacy.exists():
        return legacy
    return generic


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exported ONNX embedder/reranker models.")
    parser.add_argument("--model-dir", default=str(Path.home() / "cephalon-data" / "models"))
    parser.add_argument("--mark", action="store_true", help="Write validation result into onnx_profile.json files.")
    parser.add_argument("--reranker-batch-size", type=int, default=1, help="Explicitly validate this reranker batch size; use 1 for portable exports.")
    args = parser.parse_args()
    root = Path(args.model_dir).expanduser().resolve()
    results = {
        "embedder": validate_embedder(root / "embedder") if (root / "embedder" / "model.onnx").exists() else None,
        "reranker": validate_reranker(root / "reranker", max(1, args.reranker_batch_size)) if (root / "reranker" / "model.onnx").exists() else None,
    }
    print(json.dumps(results, indent=2))
    if args.mark and results["embedder"]:
        mark_validated(
            root / "embedder",
            "shape_runtime",
            results["embedder"]["dimension"] > 0 and np.isfinite(results["embedder"]["norm"]),
            {"dimension": results["embedder"]["dimension"], "model_sha256": model_hash(root / "embedder")},
        )
    if args.mark and results["reranker"]:
        mark_validated(
            root / "reranker",
            "single_pair_inference",
            True,
            {"score_mode": results["reranker"]["score_mode"], "max_batch_size": results["reranker"]["max_batch_size"], "model_sha256": model_hash(root / "reranker")},
        )


if __name__ == "__main__":
    main()
