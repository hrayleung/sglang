#!/usr/bin/env python3
"""
Extreme long sequence test - fill most of the KV cache with a single session.

Uses a very long input prompt to create a large unique KV cache that can be fully offloaded.
"""

import requests
import time

BASE_URL = "http://127.0.0.1:30000"

def open_session(session_id: str):
    r = requests.post(f"{BASE_URL}/open_session", json={
        "session_id": session_id,
        "capacity_of_str_len": 200000,
    }, timeout=30)
    return r.status_code == 200

def close_session(session_id: str):
    requests.post(f"{BASE_URL}/close_session", json={"session_id": session_id}, timeout=30)

def generate(session_id: str, text: str, max_tokens: int):
    r = requests.post(f"{BASE_URL}/generate", json={
        "text": text,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.8},
        "session_params": {"id": session_id},
    }, timeout=600)
    return r.json() if r.status_code == 200 else None

def tool_start(session_id: str):
    r = requests.post(f"{BASE_URL}/session/tool_start", json={
        "session_id": session_id, "mode": "cpu"
    }, timeout=60)
    return r.json() if r.status_code in [200, 400] else None

def tool_end(session_id: str, epoch: int):
    r = requests.post(f"{BASE_URL}/session/tool_end", json={
        "session_id": session_id, "epoch": epoch
    }, timeout=60)
    return r.json() if r.status_code in [200, 400] else None

def main():
    print("=" * 70)
    print("Extreme Long Sequence KV Offload Test")
    print("=" * 70)
    
    session_id = f"extreme-{int(time.time())}"
    
    # Create a long unique prompt that fits in 131K context
    # Target ~80K tokens to leave room for generation
    import random
    random.seed(int(time.time()))
    
    lines = []
    for i in range(1200):  # ~80K tokens
        rand_nums = [random.randint(1000000, 9999999) for _ in range(8)]
        lines.append(f"Row {i}: {rand_nums[0]} {rand_nums[1]} {rand_nums[2]} {rand_nums[3]} "
                    f"{rand_nums[4]} {rand_nums[5]} {rand_nums[6]} {rand_nums[7]}")
    
    long_prompt = "Data:\n" + "\n".join(lines) + "\nSummary:"
    
    print(f"Prompt length: ~{len(long_prompt)} chars")
    
    try:
        print(f"\n1. Creating session: {session_id}")
        if not open_session(session_id):
            print("Failed to create session")
            return
        
        print("\n2. Running generation with extreme long prompt...")
        print("   This will take a while to prefill...")
        start = time.time()
        result = generate(session_id, long_prompt, max_tokens=50)
        gen_time = time.time() - start
        
        if not result:
            print("Generation failed!")
            print("   The prompt may be too long for available KV cache.")
            return
        
        prompt_tokens = result.get("meta_info", {}).get("prompt_tokens", 0)
        completion_tokens = result.get("meta_info", {}).get("completion_tokens", 0)
        
        print(f"   Prompt tokens: {prompt_tokens:,}")
        print(f"   Completion tokens: {completion_tokens:,}")
        print(f"   Generation time: {gen_time:.1f}s")
        
        # Estimate KV size
        kv_estimate_mb = prompt_tokens * 82 / 1024  # ~82KB per token for 70B TP4
        print(f"   Estimated KV size: ~{kv_estimate_mb:.0f} MB")
        
        time.sleep(1)
        
        print("\n3. Calling TOOL_START (offload to CPU)...")
        print("   Watch 'free -h' or 'htop' in another terminal for CPU RAM increase")
        start = time.time()
        result = tool_start(session_id)
        offload_time = time.time() - start
        
        if not result or not result.get("ok"):
            print(f"   ✗ Offload failed: {result}")
            return
        
        epoch = result.get("epoch")
        kv_bytes = result.get("kv_bytes", 0)
        kv_mb = kv_bytes / (1024 * 1024)
        
        print(f"   ✓ Offloaded in {offload_time:.2f}s")
        print(f"   KV bytes: {kv_bytes:,} ({kv_mb:.1f} MB)")
        print(f"   Epoch: {epoch}")
        
        # Wait to observe memory
        print("\n4. Waiting 30 seconds - check CPU RAM now...")
        print("   Run: watch -n1 'free -h'")
        for i in range(30, 0, -5):
            print(f"   {i}s remaining...")
            time.sleep(5)
        
        print("\n5. Calling TOOL_END (restore to GPU)...")
        start = time.time()
        result = tool_end(session_id, epoch)
        restore_time = time.time() - start
        
        if not result or not result.get("ok"):
            print(f"   ✗ Restore failed: {result}")
            return
        
        print(f"   ✓ Restored in {restore_time:.2f}s")
        
        print("\n6. Verifying with continuation...")
        result = generate(session_id, long_prompt + " Based on the data:", max_tokens=20)
        if result:
            cached = result.get("meta_info", {}).get("cached_tokens", 0)
            print(f"   Cached tokens: {cached:,}")
            print("   ✓ Success!")
        
        print("\n" + "=" * 70)
        print(f"Test completed! Offloaded {kv_mb:.1f} MB KV cache")
        print("=" * 70)
        
    finally:
        print(f"\nCleaning up...")
        close_session(session_id)

if __name__ == "__main__":
    main()
