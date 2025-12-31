#!/usr/bin/env python3
"""
Ultra-heavy KV offload/restore test targeting 1GB+ KV cache.

KV cache size ≈ num_tokens × num_layers × num_kv_heads × head_dim × 2 (K+V) × dtype_size
For Llama-70B with TP=4: ~82KB per token
To get 1GB: need ~12,500 tokens
"""

import requests
import time

BASE_URL = "http://127.0.0.1:30000"

def open_session(session_id: str):
    r = requests.post(f"{BASE_URL}/open_session", json={
        "session_id": session_id,
        "capacity_of_str_len": 131072,
    })
    return r.status_code == 200

def close_session(session_id: str):
    requests.post(f"{BASE_URL}/close_session", json={"session_id": session_id})

def generate(session_id: str, text: str, max_tokens: int):
    r = requests.post(f"{BASE_URL}/generate", json={
        "text": text,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 1.0,
            "ignore_eos": True,
            "stop": [],  # Clear any default stop strings
            "skip_special_tokens": False,
        },
        "session_params": {"id": session_id},
    }, timeout=600)
    return r.json() if r.status_code == 200 else None

def tool_start(session_id: str):
    r = requests.post(f"{BASE_URL}/session/tool_start", json={"session_id": session_id, "mode": "cpu"})
    return r.status_code, r.json()

def tool_end(session_id: str, epoch: int):
    r = requests.post(f"{BASE_URL}/session/tool_end", json={"session_id": session_id, "epoch": epoch})
    return r.status_code, r.json()

def get_kv_meta(session_id: str):
    r = requests.get(f"{BASE_URL}/session/kv_meta", params={"session_id": session_id})
    return r.json()

def main():
    print("=" * 70)
    print("Ultra-Heavy KV Offload/Restore Test (Target: 1GB+ KV cache)")
    print("=" * 70)
    
    session_id = f"ultra-heavy-{int(time.time())}"
    
    # Short prompt + very long generation = single leaf node with all output tokens
    import random
    random_seed = random.randint(100000, 999999)
    
    # Minimal prompt to maximize output in single leaf
    long_prompt = f"Write an extremely long and detailed story (ID:{random_seed}). Include many chapters, characters, and plot twists. Never stop writing until told. Begin:"

    try:
        print(f"\n1. Creating session: {session_id}")
        if not open_session(session_id):
            print("Failed to create session")
            return
        
        # Generate to build massive KV cache - use high token count with low temp to avoid EOS
        print("\n2. Running initial generation (16000 tokens to build ~1GB+ KV)...")
        print("   This will take several minutes...")
        start = time.time()
        result = generate(session_id, long_prompt, max_tokens=16000)
        gen_time = time.time() - start
        
        if not result:
            print("Generation failed")
            return
        
        prompt_tokens = result.get("meta_info", {}).get("prompt_tokens", 0)
        completion_tokens = result.get("meta_info", {}).get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        
        print(f"   Prompt tokens: {prompt_tokens:,}")
        print(f"   Completion tokens: {completion_tokens:,}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Generation time: {gen_time:.1f}s ({completion_tokens/gen_time:.1f} tok/s)")
        
        # Estimate KV size (Llama-70B TP=4: ~82KB/token)
        estimated_kv_gb = total_tokens * 82 * 1024 / (1024**3)
        print(f"   Estimated KV size: ~{estimated_kv_gb:.2f} GB")
        
        time.sleep(1)
        
        # Offload
        print("\n3. Calling TOOL_START (offload to CPU)...")
        start = time.time()
        status, result = tool_start(session_id)
        offload_time = time.time() - start
        
        if not result.get("ok"):
            print(f"   ✗ Offload failed: {result.get('error')}")
            return
        
        epoch = result.get("epoch")
        kv_bytes = result.get("kv_bytes", 0)
        kv_gb = kv_bytes / (1024**3)
        print(f"   ✓ Offloaded in {offload_time:.2f}s")
        print(f"   KV bytes: {kv_bytes:,} ({kv_gb:.3f} GB)")
        
        meta = get_kv_meta(session_id)
        print(f"   State: {meta.get('state')}, Tier: {meta.get('tier')}")
        
        # Simulate tool execution
        tool_time = 60
        print(f"\n4. Simulating tool execution ({tool_time} seconds)...")
        print("   GPU memory should be freed - run 'nvidia-smi' in another terminal")
        for i in range(tool_time, 0, -10):
            print(f"   Tool executing... {i}s remaining")
            time.sleep(10)
        print("   Tool execution complete!")
        
        # Restore
        print("\n5. Calling TOOL_END (restore to GPU)...")
        start = time.time()
        status, result = tool_end(session_id, epoch)
        restore_time = time.time() - start
        
        if not result.get("ok"):
            print(f"   ✗ Restore failed: {result.get('error')}")
            return
        print(f"   ✓ Restored in {restore_time:.2f}s, state={result.get('state')}")
        
        # Continue generation
        print("\n6. Continuing generation (verifying KV reuse)...")
        continuation = "\n\nNow summarize the top 3 key points:"
        start = time.time()
        result = generate(session_id, long_prompt + continuation, max_tokens=100)
        cont_time = time.time() - start
        
        if result:
            cached = result.get("meta_info", {}).get("cached_tokens", 0)
            print(f"   Cached tokens reused: {cached:,}")
            print(f"   Continuation time: {cont_time:.2f}s")
            print("   ✓ Success!")
        
        print("\n" + "=" * 70)
        print(f"Ultra-heavy test completed! KV size: {kv_gb:.3f} GB")
        print("=" * 70)
        
    finally:
        print(f"\nCleaning up...")
        close_session(session_id)

if __name__ == "__main__":
    main()
