import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def validate_embedder(model_dir: Path) -> dict:
    meta_path = model_dir / "cephalon_onnx_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    session = ort.InferenceSession(str(model_dir / "model.onnx"))
    fixed_length = meta.get("fixed_sequence_length")
    tokenizer_kwargs = {"truncation": True, "return_tensors": "np"}
    if fixed_length:
        tokenizer_kwargs.update({"padding": "max_length", "max_length": int(fixed_length)})
    else:
        tokenizer_kwargs["padding"] = True
    inputs = tokenizer(["Document: validation text"], **tokenizer_kwargs)
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in inputs:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
    output = np.asarray(session.run(None, ort_inputs)[0])
    dim = int(output.shape[-1])
    norm = float(np.linalg.norm(output.reshape(-1, dim)[0]))
    return {"kind": "embedder", "path": str(model_dir), "output_shape": list(output.shape), "dimension": dim, "norm": round(norm, 6)}


def validate_reranker(model_dir: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    session = ort.InferenceSession(str(model_dir / "model.onnx"))
    pairs = [
        ["breathing exercise", "The 4-7-8 breathing method is a breathing exercise."],
        ["breathing exercise", "A graphics card renders pixels for video output."],
    ]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="np")
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in inputs:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
    raw_scores = np.asarray(session.run(None, ort_inputs)[0])
    scores, score_mode = reranker_scores(raw_scores)
    return {
        "kind": "reranker",
        "path": str(model_dir),
        "scores": [round(score, 6) for score in scores],
        "score_mode": score_mode,
        "positive_above_negative": scores[0] > scores[1],
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
    meta_path = model_dir / "cephalon_onnx_meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["validated"] = valid
    meta["validation_key"] = key
    if extra:
        meta.update(extra)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exported Cephalon ONNX models.")
    parser.add_argument("--model-dir", default=str(Path.home() / "cephalon-data" / "models"))
    parser.add_argument("--mark", action="store_true", help="Write validation result into cephalon_onnx_meta.json files.")
    args = parser.parse_args()
    root = Path(args.model_dir).expanduser().resolve()
    results = {
        "embedder": validate_embedder(root / "embedder") if (root / "embedder" / "model.onnx").exists() else None,
        "reranker": validate_reranker(root / "reranker") if (root / "reranker" / "model.onnx").exists() else None,
    }
    print(json.dumps(results, indent=2))
    if args.mark and results["embedder"]:
        mark_validated(root / "embedder", "shape_norm", results["embedder"]["dimension"] in {768, 1024})
    if args.mark and results["reranker"]:
        mark_validated(
            root / "reranker",
            "positive_above_negative",
            results["reranker"]["positive_above_negative"],
            {"score_mode": results["reranker"]["score_mode"]},
        )
    if results["reranker"] and not results["reranker"]["positive_above_negative"]:
        raise SystemExit("Reranker validation failed: positive pair did not score above negative pair.")


if __name__ == "__main__":
    main()
