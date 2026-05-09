from pathlib import Path

from backend.enterprise.ingestion import ingest_documents


def on_event(event: str, payload: dict) -> None:
    print(f"[{event}] {payload}")


if __name__ == "__main__":
    result = ingest_documents(input_dir=Path("enterprise_data/docs"), callback=on_event)
    print("Ingestion finished:")
    for k, v in result.items():
        print(f"- {k}: {v}")
