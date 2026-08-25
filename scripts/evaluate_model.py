import argparse
from pathlib import Path
import sys

import torch
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.baseline_xgboost import evaluate_baseline, select_baseline_threshold, train_baseline
from src.data.loader import build_temporal_snapshot, default_data_root, load_elliptic_data, test_period, validation_period
from src.training.hybrid_train import combine_features, evaluate_hybrid_model, extract_embeddings, select_hybrid_threshold
from src.training.train_gnn import default_checkpoint_path, load_encoder_checkpoint


default_xgboost_path = project_root / "artifacts" / "hybrid_xgboost.json"


def extract_features(encoder, data, period, num_neighbors, batch_size, device):
    snapshot = build_temporal_snapshot(data, period[1], period)
    raw_features, embeddings, labels = extract_embeddings(encoder, snapshot, num_neighbors, batch_size, device)
    return combine_features(raw_features, embeddings), labels


def evaluate_saved_model(args):
    torch.manual_seed(42)
    data = load_elliptic_data(args.data_root)
    encoder = load_encoder_checkpoint(args.encoder_checkpoint, args.device)
    model = XGBClassifier()
    model.load_model(args.xgboost_model)
    num_neighbors = tuple(args.num_neighbors)
    validation_features, validation_labels = extract_features(encoder, data, validation_period, num_neighbors, args.batch_size, args.device)
    test_features, test_labels = extract_features(encoder, data, test_period, num_neighbors, args.batch_size, args.device)
    threshold = select_hybrid_threshold(model, validation_features, validation_labels)
    hybrid_validation = evaluate_hybrid_model(model, validation_features, validation_labels, threshold)
    hybrid_test = evaluate_hybrid_model(model, test_features, test_labels, threshold)
    baseline = train_baseline(data)
    baseline_threshold = select_baseline_threshold(baseline, data)
    baseline_validation = evaluate_baseline(baseline, data, "val_mask", baseline_threshold)
    baseline_test = evaluate_baseline(baseline, data, "test_mask", baseline_threshold)
    return {"threshold": threshold, "baseline_threshold": baseline_threshold, "hybrid_validation": hybrid_validation, "hybrid_test": hybrid_test, "baseline_validation": baseline_validation, "baseline_test": baseline_test}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the saved baseline and hybrid models on validation and test periods.")
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--encoder-checkpoint", type=Path, default=default_checkpoint_path)
    parser.add_argument("--xgboost-model", type=Path, default=default_xgboost_path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", type=int, nargs=2, default=(25, 10))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    result = evaluate_saved_model(parse_args())
    print(f"Baseline threshold:                    {result['baseline_threshold']:.2f}")
    print(f"Hybrid threshold:                      {result['threshold']:.2f}")
    print(f"Baseline validation average precision: {result['baseline_validation']['average_precision']:.4f}")
    print(f"Hybrid validation average precision:   {result['hybrid_validation']['average_precision']:.4f}")
    print(f"Baseline test average precision:       {result['baseline_test']['average_precision']:.4f}")
    print(f"Hybrid test average precision:         {result['hybrid_test']['average_precision']:.4f}")
    print(f"Hybrid test precision:                 {result['hybrid_test']['precision']:.4f}")
    print(f"Hybrid test recall:                    {result['hybrid_test']['recall']:.4f}")
    print(f"Hybrid test F1:                        {result['hybrid_test']['f1']:.4f}")


if __name__ == "__main__":
    main()
