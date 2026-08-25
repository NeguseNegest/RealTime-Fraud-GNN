import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.baseline_xgboost import classification_metrics, evaluate_baseline, select_baseline_threshold, select_threshold, train_baseline
from src.data.loader import build_temporal_snapshot, default_data_root, load_elliptic_data, test_period, train_period, validation_period
from src.training.train_gnn import create_neighbor_loader, default_checkpoint_path, load_encoder_checkpoint, resolve_device


def extract_embeddings(encoder, snapshot, num_neighbors=(25, 10), batch_size=512, device="auto"):
    device = resolve_device(device)
    encoder = encoder.to(device)
    encoder.eval()
    loader = create_neighbor_loader(snapshot, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False)
    raw_batches = []
    embedding_batches = []
    label_batches = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_embeddings = encoder(batch.x, batch.edge_index)
            seed_count = batch.batch_size
            raw_batches.append(batch.x[:seed_count].cpu())
            embedding_batches.append(batch_embeddings[:seed_count].cpu())
            label_batches.append(batch.y[:seed_count].cpu())
    raw_features = torch.cat(raw_batches).numpy()
    embeddings = torch.cat(embedding_batches).numpy()
    labels = torch.cat(label_batches).numpy()
    return raw_features, embeddings, labels


def combine_features(raw_features, embeddings):
    return np.concatenate((raw_features, embeddings), axis=1)


def train_hybrid_model(features, labels, tree_depth=6):
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", max_depth=tree_depth, random_state=42, n_jobs=1)
    with threadpool_limits(limits=1):
        model.fit(features, labels)
    return model


def evaluate_hybrid_model(model, features, labels, threshold=0.5):
    probabilities = model.predict_proba(features)[:, 1]
    return classification_metrics(labels, probabilities, threshold)


def select_hybrid_threshold(model, features, labels):
    probabilities = model.predict_proba(features)[:, 1]
    return select_threshold(labels, probabilities)


def run_hybrid_training(data, encoder, num_neighbors=(25, 10), batch_size=512, tree_depth=6, device="auto"):
    torch.manual_seed(42)
    train_snapshot = build_temporal_snapshot(data, train_period[1], train_period)
    validation_snapshot = build_temporal_snapshot(data, validation_period[1], validation_period)
    test_snapshot = build_temporal_snapshot(data, test_period[1], test_period)

    train_raw, train_embeddings, train_labels = extract_embeddings(encoder, train_snapshot, num_neighbors=num_neighbors, batch_size=batch_size, device=device)
    val_raw, val_embeddings, val_labels = extract_embeddings(encoder, validation_snapshot, num_neighbors=num_neighbors, batch_size=batch_size, device=device)
    test_raw, test_embeddings, test_labels = extract_embeddings(encoder, test_snapshot, num_neighbors=num_neighbors, batch_size=batch_size, device=device)
    train_features = combine_features(train_raw, train_embeddings)
    val_features = combine_features(val_raw, val_embeddings)
    test_features = combine_features(test_raw, test_embeddings)
    hybrid_model = train_hybrid_model(train_features, train_labels, tree_depth)
    threshold = select_hybrid_threshold(hybrid_model, val_features, val_labels)
    hybrid_validation = evaluate_hybrid_model(hybrid_model, val_features, val_labels, threshold)
    hybrid_test = evaluate_hybrid_model(hybrid_model, test_features, test_labels, threshold)
    baseline_model = train_baseline(data)
    baseline_threshold = select_baseline_threshold(baseline_model, data)
    baseline_validation = evaluate_baseline(baseline_model, data, mask_name="val_mask", threshold=baseline_threshold)
    baseline_test = evaluate_baseline(baseline_model, data, mask_name="test_mask", threshold=baseline_threshold)
    return {"model": hybrid_model, "threshold": threshold, "baseline_threshold": baseline_threshold, "hybrid_validation": hybrid_validation, "hybrid_test": hybrid_test, "baseline_validation": baseline_validation, "baseline_test": baseline_test, "num_features": train_features.shape[1]}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the GraphSAGE + XGBoost hybrid AML model.")
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--encoder-checkpoint", type=Path, default=default_checkpoint_path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", type=int, nargs=2, default=(25, 10))
    parser.add_argument("--tree-depth", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_elliptic_data(args.data_root)
    encoder = load_encoder_checkpoint(args.encoder_checkpoint, args.device)
    result = run_hybrid_training(data=data, encoder=encoder, num_neighbors=tuple(args.num_neighbors), batch_size=args.batch_size, tree_depth=args.tree_depth, device=args.device)
    print(f"Hybrid feature count: {result['num_features']}")
    print(f"Validation baseline average precision: {result['baseline_validation']['average_precision']:.4f}")
    print(f"Validation hybrid average precision:   {result['hybrid_validation']['average_precision']:.4f}")
    print(f"Test baseline average precision:       {result['baseline_test']['average_precision']:.4f}")
    print(f"Test hybrid average precision:         {result['hybrid_test']['average_precision']:.4f}")
    print(f"Hybrid validation threshold:           {result['threshold']:.2f}")
    print(f"Hybrid test precision:      {result['hybrid_test']['precision']:.4f}")
    print(f"Hybrid test recall:         {result['hybrid_test']['recall']:.4f}")
    print(f"Hybrid test F1:             {result['hybrid_test']['f1']:.4f}")


if __name__ == "__main__":
    main()
