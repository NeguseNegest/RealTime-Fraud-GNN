"""Score Kafka transactions and print AML alerts."""

import argparse
import json
import os
from pathlib import Path
import sys

from kafka import KafkaConsumer
import mlflow
import numpy as np
import shap
import torch
from torch_geometric.utils import k_hop_subgraph

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import default_data_root, load_elliptic_data


default_bootstrap_servers = "localhost:9092"
default_topic = "aml-transactions"
default_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
default_model_name = "fraud-gnn-ensemble"
default_threshold = 0.85


def load_latest_model(tracking_uri, model_name):
    mlflow.set_tracking_uri(tracking_uri)
    versions = mlflow.MlflowClient().search_model_versions(f"name='{model_name}'")
    latest = max(versions, key=lambda version: int(version.version))
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest.version}")
    return model, latest.version


def create_consumer(bootstrap_servers, topic, group_id):
    return KafkaConsumer(topic, bootstrap_servers=bootstrap_servers, group_id=group_id, auto_offset_reset="earliest")


def build_model_input(data, event, edge_cache, num_hops=2):
    node_id = event["node_id"]
    time_step = event["time_step"]

    if time_step not in edge_cache:
        source, target = data.edge_index
        available = data.time_step <= time_step
        edge_cache[time_step] = data.edge_index[:, available[source] & available[target]]

    node_ids, edge_index, target_nodes, _ = k_hop_subgraph(node_id, num_hops, edge_cache[time_step], relabel_nodes=True, num_nodes=data.num_nodes)
    x = data.x[node_ids].clone()
    x[target_nodes] = torch.tensor(event["features"], dtype=torch.float32)
    return {"x": x.numpy(), "edge_index": edge_index.numpy(), "target_nodes": target_nodes.numpy()}


def hybrid_features(python_model, model_input):
    x = torch.as_tensor(model_input["x"], dtype=torch.float32)
    edge_index = torch.as_tensor(model_input["edge_index"], dtype=torch.long)
    target_nodes = torch.as_tensor(model_input["target_nodes"], dtype=torch.long)

    with torch.no_grad():
        embeddings = python_model.encoder(x, edge_index)

    return np.concatenate((x[target_nodes].numpy(), embeddings[target_nodes].numpy()), axis=1)


def feature_name(index, raw_feature_count):
    if index < raw_feature_count:
        return f"raw_feature_{index}"
    return f"embedding_{index - raw_feature_count}"


def explain_transaction(explainer, python_model, model_input):
    features = hybrid_features(python_model, model_input)
    values = explainer(features).values[0]
    indices = np.argsort(np.abs(values))[-3:][::-1]
    return [(feature_name(index, model_input["x"].shape[1]), float(values[index])) for index in indices]


def consume_transactions(data, consumer, model, threshold=default_threshold, num_hops=2, max_messages=0):
    python_model = model.unwrap_python_model()
    explainer = shap.TreeExplainer(python_model.classifier)
    edge_cache = {}
    count = 0

    for message in consumer:
        event = json.loads(message.value)
        model_input = build_model_input(data, event, edge_cache, num_hops)
        probability = float(model.predict(model_input)[0])
        count += 1
        print(f"Transaction {event['node_id']} | Fraud probability: {probability:.4f}")

        if probability > threshold:
            print(f"AML ALERT | Node {event['node_id']} | Probability {probability:.4f}")
            for name, value in explain_transaction(explainer, python_model, model_input):
                print(f"  {name}: {value:+.4f}")

        if max_messages and count >= max_messages:
            break

    return count


def parse_args():
    parser = argparse.ArgumentParser(description="Consume Kafka transactions and score them for fraud.")
    parser.add_argument("--bootstrap-servers", default=default_bootstrap_servers)
    parser.add_argument("--topic", default=default_topic)
    parser.add_argument("--group-id", default="fraud-gnn-consumer")
    parser.add_argument("--tracking-uri", default=default_tracking_uri)
    parser.add_argument("--model-name", default=default_model_name)
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--threshold", type=float, default=default_threshold)
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--max-messages", type=int, default=0, help="Number of messages to consume. Zero keeps listening.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_elliptic_data(args.data_root)
    model, version = load_latest_model(args.tracking_uri, args.model_name)
    consumer = create_consumer(args.bootstrap_servers, args.topic, args.group_id)
    print(f"Loaded {args.model_name} version {version}")

    try:
        count = consume_transactions(data, consumer, model, args.threshold, args.num_hops, args.max_messages)
    finally:
        consumer.close()

    print(f"Consumed {count} transactions")


if __name__ == "__main__":
    main()
