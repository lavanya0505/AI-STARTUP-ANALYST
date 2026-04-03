from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as memory:
    checkpoints = list(memory.list(config=None))
    print(f"Total checkpoints saved: {len(checkpoints)}")
    for cp in checkpoints[-5:]:
        thread = cp.config["configurable"]["thread_id"]
        step = cp.metadata.get("step", "?")
        source = cp.metadata.get("source", "unknown")
        print(f"  Thread: {thread[:8]}... | Step {step} | {source}")