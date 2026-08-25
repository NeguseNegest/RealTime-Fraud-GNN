import argparse
import json
import os
from pathlib import Path
import random
import sys

from kafka import KafkaConsumer
import mlflow
import numpy as np
import shap
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.streaming.producer import schema_version


default_bootstrap_servers = "localhost:9092"
default_topic = "aml-transactions"
default_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
default_model_name = "fraud-gnn-ensemble"
default_num_neighbors = (25, 10)


def load_latest_model(tracking_uri, model_name):
    mlflow.set_tracking_uri(tracking_uri)
    versions = mlflow.MlflowClient().search_model_versions(f"name='{model_name}'")
    latest = max(versions, key=lambda version: int(version.version))
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest.version}")
    return model, latest.version


def create_consumer(bootstrap_servers, topic, group_id):
    return KafkaConsumer(topic, bootstrap_servers=bootstrap_servers, group_id=group_id, auto_offset_reset="earliest")


def create_graph_state():
    return {"features": {}, "incoming_neighbors": {}}


def add_event(state, event):
    node_id = event["node_id"]
    state["features"][node_id] = torch.tensor(event["features"], dtype=torch.float32)
    state["incoming_neighbors"][node_id] = event["incoming_neighbors"]


def sample_neighborhood(state, node_id, num_neighbors=default_num_neighbors):
    generator = random.Random(node_id)
    node_ids = [node_id]
    seen_nodes = {node_id}
    edges = []
    seen_edges = set()
    frontier = [node_id]

    for limit in num_neighbors:
        next_frontier = []
        for target in frontier:
            sources = state["incoming_neighbors"][target]
            if len(sources) > limit:
                sources = generator.sample(sources, limit)
            for source in sources:
                edge = (source, target)
                if edge not in seen_edges:
                    edges.append(edge)
                    seen_edges.add(edge)
                if source not in seen_nodes:
                    node_ids.append(source)
                    seen_nodes.add(source)
                    next_frontier.append(source)
        frontier = next_frontier
    return node_ids, edges


def build_model_input(state, event, num_neighbors=default_num_neighbors):
    add_event(state, event)
    node_ids, edges = sample_neighborhood(state, event["node_id"], num_neighbors)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    x = torch.stack([state["features"][node_id] for node_id in node_ids])
    if edges:
        edge_index = torch.tensor([[node_index[source], node_index[target]] for source, target in edges], dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return {"x": x.numpy(), "edge_index": edge_index.numpy(), "target_nodes": np.array([0])}


def hybrid_features(python_model, model_input):
    x = torch.as_tensor(model_input["x"], dtype=torch.float32)
    edge_index = torch.as_tensor(model_input["edge_index"], dtype=torch.long)
    target_nodes = torch.as_tensor(model_input["target_nodes"], dtype=torch.long)
    with torch.no_grad():
        embeddings = python_model.encoder(x, edge_index)
    return np.concatenate((x[target_nodes].numpy(), embeddings[target_nodes].numpy()), axis=1)


def feature_name(index, raw_feature_count):
    if index < raw_feature_count:
        return f"local_feature_{index}"
    return f"embedding_{index - raw_feature_count}"


def explain_transaction(explainer, python_model, model_input):
    features = hybrid_features(python_model, model_input)
    values = explainer(features).values[0]
    indices = np.argsort(np.abs(values))[-3:][::-1]
    return [(feature_name(index, model_input["x"].shape[1]), float(values[index])) for index in indices]


def consume_transactions(consumer, model, threshold=None, num_neighbors=default_num_neighbors, max_messages=0):
    python_model = model.unwrap_python_model()
    threshold = python_model.threshold if threshold is None else threshold
    explainer = shap.TreeExplainer(python_model.classifier)
    state = create_graph_state()
    count = 0

    for message in consumer:
        event = json.loads(message.value)
        if event.get("schema_version") != schema_version:
            continue
        model_input = build_model_input(state, event, num_neighbors)
        score = float(model.predict(model_input)[0])
        count += 1
        print(f"Transaction {event['node_id']} | Illicit transaction score: {score:.4f}")
        if score > threshold:
            print(f"AML REVIEW | Node {event['node_id']} | Score {score:.4f}")
            for name, value in explain_transaction(explainer, python_model, model_input):
                print(f"  {name}: {value:+.4f}")
        if max_messages and count >= max_messages:
            break
    return count


def parse_args():
    parser = argparse.ArgumentParser(description="Consume Kafka transactions and score them for illicit activity.")
    parser.add_argument("--bootstrap-servers", default=default_bootstrap_servers)
    parser.add_argument("--topic", default=default_topic)
    parser.add_argument("--group-id", default="fraud-gnn-consumer")
    parser.add_argument("--tracking-uri", default=default_tracking_uri)
    parser.add_argument("--model-name", default=default_model_name)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--num-neighbors", type=int, nargs=2, default=default_num_neighbors)
    parser.add_argument("--max-messages", type=int, default=0, help="Number of messages to consume. Zero keeps listening.")
    return parser.parse_args()


def main():
    args = parse_args()
    model, version = load_latest_model(args.tracking_uri, args.model_name)
    consumer = create_consumer(args.bootstrap_servers, args.topic, args.group_id)
    threshold = model.unwrap_python_model().threshold if args.threshold is None else args.threshold
    print(f"Loaded {args.model_name} version {version} with threshold {threshold:.2f}")
    try:
        count = consume_transactions(consumer, model, threshold, tuple(args.num_neighbors), args.max_messages)
    finally:
        consumer.close()
    print(f"Consumed {count} transactions")


if __name__ == "__main__":
    main()
