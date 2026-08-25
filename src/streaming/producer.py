import argparse
import heapq
import json
from pathlib import Path
import sys
import time

from kafka import KafkaProducer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import default_data_root, load_elliptic_data, test_period


default_topic = "aml-transactions"
default_bootstrap_servers = "localhost:9092"
schema_version = 2


def create_producer(bootstrap_servers):
    return KafkaProducer(bootstrap_servers=bootstrap_servers)


def prepare_test_stream(data, limit=0):
    start, end = test_period
    node_ids = ((data.time_step >= start) & (data.time_step <= end)).nonzero(as_tuple=False).view(-1).tolist()
    selected = set(node_ids)
    incoming_neighbors = {node_id: [] for node_id in node_ids}
    outgoing_neighbors = {node_id: [] for node_id in node_ids}
    source_nodes, target_nodes = data.edge_index.tolist()

    for source, target in zip(source_nodes, target_nodes):
        if source in selected and target in selected:
            incoming_neighbors[target].append(source)
            outgoing_neighbors[source].append(target)

    indegree = {node_id: len(incoming_neighbors[node_id]) for node_id in node_ids}
    queue = [(int(data.time_step[node_id]), node_id) for node_id in node_ids if indegree[node_id] == 0]
    heapq.heapify(queue)
    ordered = []

    while queue:
        _, node_id = heapq.heappop(queue)
        ordered.append(node_id)
        for target in outgoing_neighbors[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(queue, (int(data.time_step[target]), target))

    if limit:
        ordered = ordered[:limit]
    return ordered, incoming_neighbors


def get_test_node_ids(data, limit=0):
    node_ids, _ = prepare_test_stream(data, limit)
    return node_ids


def build_event(data, node_id, incoming_neighbors):
    return {"schema_version": schema_version, "node_id": node_id, "time_step": int(data.time_step[node_id]), "features": data.x[node_id].tolist(), "incoming_neighbors": incoming_neighbors[node_id]}


def publish_transactions(data, producer, topic=default_topic, delay=0.01, limit=0):
    node_ids, incoming_neighbors = prepare_test_stream(data, limit)
    started = time.perf_counter()

    for node_id in node_ids:
        event = build_event(data, node_id, incoming_neighbors)
        producer.send(topic, key=str(node_id).encode("utf-8"), value=json.dumps(event).encode("utf-8"))
        time.sleep(delay)

    producer.flush()
    elapsed = time.perf_counter() - started
    return len(node_ids), elapsed


def parse_args():
    parser = argparse.ArgumentParser(description="Publish Elliptic test transactions to Kafka.")
    parser.add_argument("--bootstrap-servers", default=default_bootstrap_servers)
    parser.add_argument("--topic", default=default_topic)
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--delay", type=float, default=0.01, help="Seconds to wait between transactions.")
    parser.add_argument("--limit", type=int, default=0, help="Number of transactions to publish. Zero publishes all test transactions.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_elliptic_data(args.data_root)
    producer = create_producer(args.bootstrap_servers)

    try:
        count, elapsed = publish_transactions(data, producer, args.topic, args.delay, args.limit)
    finally:
        producer.close()

    print(f"Published {count} transactions to {args.topic} in {elapsed:.2f} seconds")
    print(f"Throughput: {count / elapsed:.2f} messages per second")


if __name__ == "__main__":
    main()
