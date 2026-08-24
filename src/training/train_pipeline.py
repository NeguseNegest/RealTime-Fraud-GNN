"""Train and register the GraphSAGE and XGBoost ensemble."""

import argparse
import os
from pathlib import Path
import sys

import mlflow

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import default_data_root, load_elliptic_data
from src.model.ensemble import FraudEnsemble
from src.training.hybrid_train import run_hybrid_training
from src.training.train_gnn import default_checkpoint_path, save_encoder_checkpoint, train_graphsage


project_root = Path(__file__).resolve().parents[2]
default_xgboost_path = project_root / "artifacts" / "hybrid_xgboost.json"
default_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")


def run_training_pipeline(args):
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    data = load_elliptic_data(args.data_root)
    parameters = {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "tree_depth": args.tree_depth, "hop_1_neighbors": args.num_neighbors[0], "hop_2_neighbors": args.num_neighbors[1], "device": args.device}

    with mlflow.start_run() as run:
        mlflow.log_params(parameters)
        gnn_result = train_graphsage(data, epochs=args.epochs, batch_size=args.batch_size, num_neighbors=tuple(args.num_neighbors), learning_rate=args.learning_rate, device=args.device)
        encoder_path = save_encoder_checkpoint(gnn_result["model"].encoder, args.encoder_checkpoint)
        hybrid_result = run_hybrid_training(data, gnn_result["model"].encoder, num_neighbors=tuple(args.num_neighbors), batch_size=args.batch_size, tree_depth=args.tree_depth, device=args.device)
        args.xgboost_model.parent.mkdir(parents=True, exist_ok=True)
        hybrid_result["model"].save_model(str(args.xgboost_model))
        metrics = {"gnn_validation_pr_auc": gnn_result["best_val_pr_auc"], "gnn_best_epoch": gnn_result["best_epoch"], "baseline_validation_pr_auc": hybrid_result["baseline_validation"]["pr_auc"], "baseline_test_pr_auc": hybrid_result["baseline_test"]["pr_auc"], "hybrid_validation_pr_auc": hybrid_result["hybrid_validation"]["pr_auc"], "hybrid_test_pr_auc": hybrid_result["hybrid_test"]["pr_auc"], "hybrid_test_precision": hybrid_result["hybrid_test"]["precision"], "hybrid_test_recall": hybrid_result["hybrid_test"]["recall"], "hybrid_test_f1": hybrid_result["hybrid_test"]["f1"]}
        mlflow.log_metrics(metrics)
        model_info = mlflow.pyfunc.log_model(name="ensemble", python_model=FraudEnsemble(), artifacts={"encoder": str(encoder_path.resolve()), "xgboost": str(args.xgboost_model.resolve())}, code_paths=[str(project_root / "src")], pip_requirements=str(project_root / "requirements.txt"), registered_model_name=args.registered_model_name)

    return {"run_id": run.info.run_id, "model_uri": model_info.model_uri, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description="Train and register the fraud ensemble with MLflow.")
    parser.add_argument("--tracking-uri", default=default_tracking_uri)
    parser.add_argument("--experiment-name", default="fraud-gnn")
    parser.add_argument("--registered-model-name", default="fraud-gnn-ensemble")
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--encoder-checkpoint", type=Path, default=default_checkpoint_path)
    parser.add_argument("--xgboost-model", type=Path, default=default_xgboost_path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", type=int, nargs=2, default=(25, 10))
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--tree-depth", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    result = run_training_pipeline(parse_args())
    print(f"Run ID: {result['run_id']}")
    print(f"Model URI: {result['model_uri']}")
    print(f"Validation PR-AUC: {result['metrics']['hybrid_validation_pr_auc']:.4f}")
    print(f"Test PR-AUC: {result['metrics']['hybrid_test_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
