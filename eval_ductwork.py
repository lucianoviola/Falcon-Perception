"""Evaluate Falcon-Perception on aerial ductwork detection.

Runs zero-shot detection on the ductwork OD test set (640x640 tiles)
and computes mAP@50, precision, recall, F1 against COCO ground truth.

Usage:
    uv run python eval_ductwork.py
    uv run python eval_ductwork.py --limit 20   # quick test
    uv run python eval_ductwork.py --query "ductwork on rooftop"
    uv run python eval_ductwork.py --visualize   # save annotated images
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


# ── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).resolve().parent.parent / "ductwork-detector" / "data"
OD_DATASET = DATA_ROOT / "od_dataset" / "tiles_640_overlap_160_best_tile"
TEST_DIR = OD_DATASET / "test"
ANNOTATIONS_FILE = TEST_DIR / "annotations.json"
IMAGES_DIR = TEST_DIR / "images"


def load_coco_annotations(path: Path) -> tuple[list[dict], list[dict], dict[int, str]]:
    """Load COCO annotations. Returns (images, annotations, cat_id->name)."""
    with open(path) as f:
        data = json.load(f)
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    return data["images"], data["annotations"], cat_map


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def falcon_bbox_to_xyxy(bbox: dict, img_w: int, img_h: int) -> np.ndarray:
    """Convert Falcon {x,y,h,w} (center, normalized) to [x1,y1,x2,y2] pixels."""
    cx = bbox["x"] * img_w
    cy = bbox["y"] * img_h
    bw = bbox["w"] * img_w
    bh = bbox["h"] * img_h
    return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])


def coco_bbox_to_xyxy(bbox: list[float]) -> np.ndarray:
    """Convert COCO [x, y, w, h] (top-left) to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h])


def pair_bbox_entries(raw: list[dict]) -> list[dict]:
    """Pair [{x,y}, {h,w}, ...] into [{x,y,h,w}, ...]."""
    bboxes, current = [], {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        current.update(entry)
        if all(k in current for k in ("x", "y", "h", "w")):
            bboxes.append(dict(current))
            current = {}
    return bboxes


def match_predictions(
    pred_boxes: list[np.ndarray],
    gt_boxes: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Match predictions to GT. Returns (TP, FP, FN)."""
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def run_evaluation(
    query: str = "ductwork",
    limit: int = -1,
    iou_threshold: float = 0.5,
    visualize: bool = False,
    out_dir: str = "./outputs/ductwork_eval",
    queries: list[str] | None = None,
) -> None:
    """Run the full evaluation pipeline."""
    print("=" * 60)
    print("Falcon-Perception → Ductwork Detection Eval")
    print("=" * 60)

    # Load GT
    images, annotations, cat_map = load_coco_annotations(ANNOTATIONS_FILE)
    print(f"GT: {len(images)} images, {len(annotations)} annotations")
    print(f"Categories: {cat_map}")

    # Build per-image GT lookup
    img_id_to_info = {img["id"]: img for img in images}
    gt_by_image: dict[int, list[list[float]]] = {}
    for ann in annotations:
        gt_by_image.setdefault(ann["image_id"], []).append(ann["bbox"])

    if limit > 0:
        images = images[:limit]
    print(f"Evaluating {len(images)} images")

    # All queries to try
    all_queries = queries or [query]
    print(f"Queries: {all_queries}\n")

    # Load model (once)
    print("Loading Falcon-Perception (MLX) ...")
    t0 = time.perf_counter()
    from falcon_perception import (
        PERCEPTION_MODEL_ID,
        build_prompt_for_task,
        load_and_prepare_model,
    )
    from falcon_perception.mlx.batch_inference import (
        BatchInferenceEngine,
        process_batch_and_generate,
    )

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=PERCEPTION_MODEL_ID,
        dtype="float16",
        backend="mlx",
    )
    engine = BatchInferenceEngine(model, tokenizer)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    for q in all_queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {q!r}")
        print(f"{'─' * 60}")

        prompt = build_prompt_for_task(q, "detection")

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_pred = 0
        total_gt = 0
        timings: list[float] = []
        per_image_results: list[dict] = []

        vis_dir = Path(out_dir) / q.replace(" ", "_")
        if visualize:
            vis_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_info in enumerate(images):
            img_path = IMAGES_DIR / img_info["file_name"]
            if not img_path.exists():
                print(f"  SKIP {img_info['file_name']} (not found)")
                continue

            pil_image = Image.open(img_path).convert("RGB")
            w, h = pil_image.size

            # Run inference
            t0 = time.perf_counter()
            batch = process_batch_and_generate(
                tokenizer,
                [(pil_image, prompt)],
                max_length=model_args.max_seq_len,
                min_dimension=256,
                max_dimension=1024,
            )
            _, aux_outputs = engine.generate(
                tokens=batch["tokens"],
                pos_t=batch["pos_t"],
                pos_hw=batch["pos_hw"],
                pixel_values=batch["pixel_values"],
                pixel_mask=batch["pixel_mask"],
                max_new_tokens=200,
                temperature=0.0,
                task="detection",
            )
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)

            # Parse predictions
            aux = aux_outputs[0]
            bboxes = pair_bbox_entries(aux.bboxes_raw)
            pred_boxes = [falcon_bbox_to_xyxy(b, w, h) for b in bboxes]

            # GT boxes for this image
            gt_coco = gt_by_image.get(img_info["id"], [])
            gt_boxes = [coco_bbox_to_xyxy(b) for b in gt_coco]

            # Match
            tp, fp, fn = match_predictions(pred_boxes, gt_boxes, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_pred += len(pred_boxes)
            total_gt += len(gt_boxes)

            per_image_results.append({
                "file": img_info["file_name"],
                "pred": len(pred_boxes),
                "gt": len(gt_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "time": elapsed,
            })

            status = "OK" if (fp == 0 and fn == 0) else ""
            if fp > 0:
                status += f" FP={fp}"
            if fn > 0:
                status += f" FN={fn}"

            if (idx + 1) % 25 == 0 or idx == len(images) - 1:
                print(f"  [{idx+1}/{len(images)}] avg={np.mean(timings):.2f}s/img")

            # Visualize
            if visualize and (fp > 0 or fn > 0 or len(pred_boxes) > 0):
                from PIL import ImageDraw

                vis = pil_image.copy()
                draw = ImageDraw.Draw(vis)
                # GT in green
                for box in gt_boxes:
                    draw.rectangle(box.tolist(), outline="green", width=3)
                # Pred in red
                for box in pred_boxes:
                    draw.rectangle(box.tolist(), outline="red", width=2)
                vis.save(vis_dir / f"{img_info['file_name']}")

        # Aggregate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{'=' * 60}")
        print(f"Results for query: {q!r}  (IoU={iou_threshold})")
        print(f"{'=' * 60}")
        print(f"  Images evaluated : {len(per_image_results)}")
        print(f"  GT annotations   : {total_gt}")
        print(f"  Predictions      : {total_pred}")
        print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
        print(f"  Precision : {precision:.3f}")
        print(f"  Recall    : {recall:.3f}")
        print(f"  F1        : {f1:.3f}")
        print(f"  Avg time  : {np.mean(timings):.2f}s/image")
        print(f"  Total time: {sum(timings):.1f}s")

        # Error analysis
        fp_images = [r for r in per_image_results if r["fp"] > 0]
        fn_images = [r for r in per_image_results if r["fn"] > 0]
        print(f"\n  Images with FP: {len(fp_images)}")
        print(f"  Images with FN: {len(fn_images)}")

        if visualize:
            print(f"  Visualizations: {vis_dir}")

    # Save results
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "queries": all_queries,
                "iou_threshold": iou_threshold,
                "n_images": len(per_image_results),
                "total_gt": total_gt,
                "total_pred": total_pred,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "per_image": per_image_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Falcon-Perception on ductwork detection")
    parser.add_argument("--query", type=str, default="ductwork", help="Detection query")
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=None,
        help="Multiple queries to evaluate (overrides --query)",
    )
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of images (-1 = all)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--visualize", action="store_true", help="Save annotated images")
    parser.add_argument("--out-dir", type=str, default="./outputs/ductwork_eval")
    args = parser.parse_args()

    run_evaluation(
        query=args.query,
        limit=args.limit,
        iou_threshold=args.iou_threshold,
        visualize=args.visualize,
        out_dir=args.out_dir,
        queries=args.queries,
    )


if __name__ == "__main__":
    main()
