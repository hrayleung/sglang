#!/usr/bin/env python3
"""
Heavy KV offload/restore test with larger context and longer tool execution.
"""

import requests
import time
import json

BASE_URL = "http://127.0.0.1:30000"

def check_server():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

def open_session(session_id: str):
    r = requests.post(f"{BASE_URL}/open_session", json={
        "session_id": session_id,
        "capacity_of_str_len": 131072,
    })
    return r.status_code == 200

def close_session(session_id: str):
    requests.post(f"{BASE_URL}/close_session", json={"session_id": session_id})

def generate(session_id: str, text: str, max_tokens: int = 50):
    r = requests.post(f"{BASE_URL}/generate", json={
        "text": text,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.7},
        "session_params": {"id": session_id},
    })
    return r.json() if r.status_code == 200 else None

def tool_start(session_id: str):
    r = requests.post(f"{BASE_URL}/session/tool_start", json={"session_id": session_id, "mode": "cpu"})
    return r.status_code, r.json()

def tool_end(session_id: str, epoch: int, tool_result: str = None):
    payload = {"session_id": session_id, "epoch": epoch}
    if tool_result:
        payload["tool_result"] = tool_result
    r = requests.post(f"{BASE_URL}/session/tool_end", json=payload)
    return r.status_code, r.json()

def get_kv_meta(session_id: str):
    r = requests.get(f"{BASE_URL}/session/kv_meta", params={"session_id": session_id})
    return r.json()

def main():
    print("=" * 70)
    print("Heavy KV Offload/Restore Test")
    print("=" * 70)
    
    if not check_server():
        print("Server not available!")
        return
    
    session_id = f"heavy-test-{int(time.time())}"
    
    # Long prompt to create larger KV cache
    long_prompt = """You are a helpful AI assistant. I need you to help me with a complex task.

Here is some context about the project:
- We are building a distributed system for processing large-scale data
- The system needs to handle millions of requests per second
- We use Kubernetes for orchestration and Redis for caching
- The backend is written in Python with FastAPI
- We have multiple microservices communicating via gRPC

The current architecture has the following components:
1. API Gateway - handles authentication and rate limiting
2. Request Router - distributes requests to appropriate services
3. Processing Workers - perform the actual computation
4. Result Aggregator - combines results from multiple workers
5. Cache Layer - stores frequently accessed data
6. Database Cluster - persistent storage with replication

We are experiencing performance issues with the following symptoms:
- High latency during peak hours (>500ms p99)
- Memory usage spikes on processing workers
- Occasional timeouts on database queries
- Cache hit ratio dropping below 80%

Please analyze this situation and provide recommendations. First, let me give you more details about our monitoring data:

CPU utilization averages 70% across workers, with spikes to 95% during batch processing.
Memory usage is around 12GB per worker out of 16GB available.
Network bandwidth is at 60% capacity with occasional congestion.
Database connection pool is often exhausted during peak times.

Now I need you to think step by step about potential solutions."""

    try:
        print(f"\n1. Creating session: {session_id}")
        if not open_session(session_id):
            print("Failed to create session")
            return
        
        # Generate a long response to build up KV cache
        print("\n2. Running initial generation (500 tokens to build large KV)...")
        start = time.time()
        result = generate(session_id, long_prompt, max_tokens=500)
        gen_time = time.time() - start
        if not result:
            print("Generation failed")
            return
        
        initial_output = result.get("text", "")
        prompt_tokens = result.get("meta_info", {}).get("prompt_tokens", 0)
        completion_tokens = result.get("meta_info", {}).get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        
        print(f"   Prompt tokens: {prompt_tokens}")
        print(f"   Completion tokens: {completion_tokens}")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Generation time: {gen_time:.2f}s")
        print(f"   Output preview: {initial_output[:200]}...")
        
        time.sleep(0.5)
        
        # Offload
        print("\n3. Calling TOOL_START (offload to CPU)...")
        status, result = tool_start(session_id)
        if not result.get("ok"):
            print(f"   ✗ Offload failed: {result.get('error')}")
            return
        
        epoch = result.get("epoch")
        kv_bytes = result.get("kv_bytes", 0)
        print(f"   ✓ Offloaded! epoch={epoch}")
        print(f"   KV bytes: {kv_bytes:,} ({kv_bytes/1024/1024:.2f} MB)")
        
        # Check meta
        meta = get_kv_meta(session_id)
        print(f"   State: {meta.get('state')}, Tier: {meta.get('tier')}")
        
        # Simulate long tool execution
        tool_time = 30
        print(f"\n4. Simulating tool execution ({tool_time} seconds)...")
        print("   (GPU memory should be freed - check nvidia-smi or server logs)")
        for i in range(tool_time, 0, -5):
            print(f"   Tool executing... {i}s remaining")
            time.sleep(5)
        print("   Tool execution complete!")
        
        # Restore
        print("\n5. Calling TOOL_END (restore to GPU)...")
        status, result = tool_end(session_id, epoch, tool_result="Tool returned: optimization recommendations generated")
        if not result.get("ok"):
            print(f"   ✗ Restore failed: {result.get('error')}")
            return
        print(f"   ✓ Restored! state={result.get('state')}")
        
        # Check meta after restore
        meta = get_kv_meta(session_id)
        print(f"   State: {meta.get('state')}, Tier: {meta.get('tier')}")
        
        # Continue generation
        print("\n6. Continuing generation (should reuse restored KV)...")
        continuation = initial_output + "\n\nBased on the analysis above, what are the top 3 priorities?"
        start = time.time()
        result = generate(session_id, long_prompt + continuation, max_tokens=200)
        cont_time = time.time() - start
        
        if result:
            cached = result.get("meta_info", {}).get("cached_tokens", 0)
            print(f"   Cached tokens reused: {cached}")
            print(f"   Continuation time: {cont_time:.2f}s")
            print(f"   Output preview: {result.get('text', '')[:200]}...")
            print("   ✓ Generation continued successfully!")
        else:
            print("   ✗ Continuation failed")
        
        print("\n" + "=" * 70)
        print("Heavy test completed!")
        print("=" * 70)
        
    finally:
        print(f"\nCleaning up session {session_id}...")
        close_session(session_id)

if __name__ == "__main__":
    main()
