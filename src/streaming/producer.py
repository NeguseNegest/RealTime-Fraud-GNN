import argparse
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


def create_producer(bootstrap_servers):
    return KafkaProducer(bootstrap_servers=bootstrap_servers)


def get_test_node_ids(data, limit=0):
    start, end = test_period
    node_ids = ((data.time_step >= start) & (data.time_step <= end)).nonzero(as_tuple=False).view(-1).tolist()
    node_ids.sort(key=lambda node_id: (int(data.time_step[node_id]), node_id))
    if limit:
        return node_ids[:limit]
    return node_ids


def build_event(data, node_id):
    return {"node_id": node_id, "time_step": int(data.time_step[node_id]), "features": data.x[node_id].tolist()}


def publish_transactions(data, producer, topic=default_topic, delay=0.01, limit=0):
    node_ids = get_test_node_ids(data, limit)
    started = time.perf_counter()

    for node_id in node_ids:
        event = build_event(data, node_id)
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
