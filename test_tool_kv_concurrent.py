#!/usr/bin/env python3
"""
Concurrent KV offload/restore stress test.

Simulates multiple sessions with random tool_start/tool_end sequences.
"""

import requests
import time
import random
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://127.0.0.1:30000"

class SessionState:
    def __init__(self, session_id):
        self.session_id = session_id
        self.epoch = 0
        self.offloaded = False
        self.lock = threading.Lock()

def open_session(session_id: str):
    r = requests.post(f"{BASE_URL}/open_session", json={
        "session_id": session_id,
        "capacity_of_str_len": 131072,
    }, timeout=30)
    return r.status_code == 200

def close_session(session_id: str):
    requests.post(f"{BASE_URL}/close_session", json={"session_id": session_id}, timeout=30)

def generate(session_id: str, text: str, max_tokens: int = 100):
    r = requests.post(f"{BASE_URL}/generate", json={
        "text": text,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.8},
        "session_params": {"id": session_id},
    }, timeout=60)
    return r.json() if r.status_code == 200 else None

def tool_start(session_id: str):
    r = requests.post(f"{BASE_URL}/session/tool_start", json={
        "session_id": session_id, "mode": "cpu"
    }, timeout=30)
    return r.json() if r.status_code in [200, 400] else None

def tool_end(session_id: str, epoch: int):
    r = requests.post(f"{BASE_URL}/session/tool_end", json={
        "session_id": session_id, "epoch": epoch
    }, timeout=30)
    return r.json() if r.status_code in [200, 400] else None

def session_worker(state: SessionState, num_cycles: int, results: list):
    """Worker that runs multiple generate/offload/restore cycles for a session."""
    session_id = state.session_id
    prompts = [
        f"Session {session_id}: Explain quantum computing in detail.",
        f"Session {session_id}: Write a poem about artificial intelligence.",
        f"Session {session_id}: Describe the history of the internet.",
        f"Session {session_id}: What are the benefits of renewable energy?",
        f"Session {session_id}: Explain machine learning algorithms.",
    ]
    
    success_count = 0
    error_count = 0
    
    for cycle in range(num_cycles):
        try:
            # Random delay to create interleaving
            time.sleep(random.uniform(0.1, 0.5))
            
            # Generate some tokens
            prompt = random.choice(prompts) + f" (cycle {cycle})"
            result = generate(session_id, prompt, max_tokens=random.randint(50, 150))
            if not result:
                error_count += 1
                continue
            
            # Random delay
            time.sleep(random.uniform(0.1, 0.3))
            
            # Offload to CPU
            with state.lock:
                offload_result = tool_start(session_id)
                if offload_result and offload_result.get("ok"):
                    state.epoch = offload_result.get("epoch", 0)
                    state.offloaded = True
                    kv_bytes = offload_result.get("kv_bytes", 0)
                else:
                    error_count += 1
                    continue
            
            # Simulate tool execution
            time.sleep(random.uniform(0.5, 2.0))
            
            # Restore from CPU
            with state.lock:
                if state.offloaded:
                    restore_result = tool_end(session_id, state.epoch)
                    if restore_result and restore_result.get("ok"):
                        state.offloaded = False
                        success_count += 1
                    else:
                        error_count += 1
                        
        except Exception as e:
            error_count += 1
    
    results.append({
        "session_id": session_id,
        "success": success_count,
        "errors": error_count,
    })

def main():
    print("=" * 70)
    print("Concurrent KV Offload/Restore Stress Test")
    print("=" * 70)
    
    num_sessions = 5
    cycles_per_session = 10
    
    print(f"\nConfig: {num_sessions} sessions × {cycles_per_session} cycles each")
    print(f"Total operations: {num_sessions * cycles_per_session} offload/restore cycles\n")
    
    # Create sessions
    sessions = []
    for i in range(num_sessions):
        session_id = f"stress-{int(time.time())}-{i}"
        if open_session(session_id):
            sessions.append(SessionState(session_id))
            print(f"Created session: {session_id}")
        else:
            print(f"Failed to create session {i}")
    
    if not sessions:
        print("No sessions created!")
        return
    
    print(f"\nStarting {len(sessions)} concurrent workers...")
    start_time = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=num_sessions) as executor:
        futures = [
            executor.submit(session_worker, state, cycles_per_session, results)
            for state in sessions
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker error: {e}")
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    
    total_success = 0
    total_errors = 0
    for r in results:
        print(f"  {r['session_id']}: {r['success']} success, {r['errors']} errors")
        total_success += r['success']
        total_errors += r['errors']
    
    print(f"\nTotal: {total_success} successful cycles, {total_errors} errors")
    print(f"Time: {elapsed:.1f}s")
    print(f"Throughput: {total_success / elapsed:.2f} cycles/sec")
    
    # Cleanup
    print("\nCleaning up sessions...")
    for state in sessions:
        close_session(state.session_id)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
