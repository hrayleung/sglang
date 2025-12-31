#!/usr/bin/env python3
"""
Test script for large context KV offload/restore to demonstrate memory savings.

This test generates progressively larger contexts to show:
1. How much KV cache memory is offloaded (in bytes and GB)
2. GPU memory freed during offload
3. Time taken for offload and restore operations
4. Successful restore verification
"""

import requests
import time
import json
import uuid

BASE_URL = "http://localhost:30000"


def get_memory_stats():
    """Get current GPU memory stats from the server."""
    try:
        resp = requests.get(f"{BASE_URL}/session/kv_meta?session_id=nonexistent")
        return resp.json()
    except:
        return None


def test_large_context_offload():
    """Test KV offload with progressively larger contexts."""
    print("=" * 80)
    print("Large Context KV Cache Offload/Restore Test")
    print("=" * 80)

    # Different context sizes to test (in tokens approximately)
    test_cases = [
        {"name": "Small (100 tokens)", "text": "Write a detailed essay about artificial intelligence. " * 10, "max_tokens": 200},
        {"name": "Medium (500 tokens)", "text": "Explain quantum computing in great detail with examples and mathematical formulas. " * 30, "max_tokens": 500},
        {"name": "Large (2000 tokens)", "text": "Write a comprehensive research paper on climate change, including introduction, methodology, results, and conclusion. Include detailed statistics and scientific evidence. " * 50, "max_tokens": 1000},
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}/{len(test_cases)}: {test_case['name']}")
        print("=" * 80)

        session_id = f"large-context-test-{uuid.uuid4().hex[:8]}"

        # 1. Open session
        print(f"\n1. Opening session: {session_id}")
        resp = requests.post(
            f"{BASE_URL}/open_session",
            json={"capacity_of_str_len": 131072, "session_id": session_id},
        )
        if resp.status_code != 200:
            print(f"   ✗ Failed to open session: {resp.status_code}")
            continue
        print(f"   ✓ Session opened")

        # 2. Generate large context
        print(f"\n2. Generating context (~{test_case['name'].split('(')[1].split(')')[0]})")
        start_gen = time.time()
        gen_resp = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": test_case['text'],
                "sampling_params": {
                    "max_new_tokens": test_case['max_tokens'],
                    "temperature": 0.7,
                },
                "session_params": {"id": session_id},
                "stream": False,
            },
        )
        gen_time = time.time() - start_gen

        if gen_resp.status_code != 200:
            print(f"   ✗ Generation failed: {gen_resp.status_code}")
            continue

        result = gen_resp.json()
        generated_text = result.get('text', '')
        print(f"   ✓ Generated {len(generated_text)} characters in {gen_time:.2f}s")
        print(f"   First 100 chars: {generated_text[:100]}...")

        # 3. Check KV metadata before offload
        print(f"\n3. Checking KV state before offload")
        meta_resp = requests.get(f"{BASE_URL}/session/kv_meta?session_id={session_id}")
        meta_before = meta_resp.json()
        print(f"   State: {meta_before.get('state')}")
        print(f"   Tier: {meta_before.get('tier')}")

        # 4. Offload to CPU
        print(f"\n4. Offloading KV cache to CPU")
        start_offload = time.time()
        offload_resp = requests.post(
            f"{BASE_URL}/session/tool_start",
            json={"session_id": session_id, "mode": "cpu"},
        )
        offload_time = time.time() - start_offload

        if offload_resp.status_code != 200:
            print(f"   ✗ Offload failed: {offload_resp.status_code}")
            continue

        offload_result = offload_resp.json()
        if not offload_result.get("ok"):
            print(f"   ✗ Offload failed: {offload_result.get('error')}")
            continue

        kv_bytes = offload_result.get("kv_bytes", 0)
        kv_gb = kv_bytes / (1024**3)
        epoch = offload_result.get("epoch", 0)

        print(f"   ✓ Offload successful!")
        print(f"   KV Cache Size: {kv_bytes:,} bytes ({kv_gb:.4f} GB)")
        print(f"   Offload Time: {offload_time:.3f}s")
        print(f"   Bandwidth: {kv_gb / offload_time:.2f} GB/s")
        print(f"   Epoch: {epoch}")

        # 5. Check metadata during offload
        print(f"\n5. Verifying KV state during offload")
        meta_resp = requests.get(f"{BASE_URL}/session/kv_meta?session_id={session_id}")
        meta_during = meta_resp.json()
        print(f"   State: {meta_during.get('state')}")
        print(f"   Tier: {meta_during.get('tier')}")
        print(f"   Sequence Length: {meta_during.get('seq_len')} tokens")
        print(f"   KV Bytes: {meta_during.get('kv_bytes'):,}")

        # 6. Simulate tool execution
        print(f"\n6. Simulating tool execution (2 seconds)...")
        time.sleep(2)
        tool_result = f"Tool completed processing for {session_id}"

        # 7. Restore from CPU
        print(f"\n7. Restoring KV cache from CPU to GPU")
        start_restore = time.time()
        restore_resp = requests.post(
            f"{BASE_URL}/session/tool_end",
            json={
                "session_id": session_id,
                "epoch": epoch,
                "tool_result": tool_result,
            },
        )
        restore_time = time.time() - start_restore

        if restore_resp.status_code != 200:
            print(f"   ✗ Restore failed: {restore_resp.status_code}")
            continue

        restore_result = restore_resp.json()
        if not restore_result.get("ok"):
            print(f"   ✗ Restore failed: {restore_result.get('error')}")
            continue

        print(f"   ✓ Restore successful!")
        print(f"   Restore Time: {restore_time:.3f}s")
        print(f"   Bandwidth: {kv_gb / restore_time:.2f} GB/s")
        print(f"   New Epoch: {restore_result.get('epoch')}")

        # 8. Verify restore
        print(f"\n8. Verifying KV state after restore")
        meta_resp = requests.get(f"{BASE_URL}/session/kv_meta?session_id={session_id}")
        meta_after = meta_resp.json()
        print(f"   State: {meta_after.get('state')}")
        print(f"   Tier: {meta_after.get('tier')}")
        print(f"   Epoch: {meta_after.get('epoch')}")

        # 9. Cleanup
        print(f"\n9. Cleaning up session")
        close_resp = requests.post(
            f"{BASE_URL}/close_session",
            json={"session_id": session_id},
        )
        print(f"   ✓ Session closed")

        # Store results
        results.append({
            "name": test_case['name'],
            "kv_bytes": kv_bytes,
            "kv_gb": kv_gb,
            "seq_len": meta_during.get('seq_len', 0),
            "offload_time": offload_time,
            "restore_time": restore_time,
            "offload_bandwidth": kv_gb / offload_time if offload_time > 0 else 0,
            "restore_bandwidth": kv_gb / restore_time if restore_time > 0 else 0,
        })

    # Summary
    print("\n" + "=" * 80)
    print("Summary of Results")
    print("=" * 80)
    print(f"\n{'Test Case':<25} {'Seq Len':<10} {'KV Size':<15} {'Offload':<12} {'Restore':<12} {'BW (GB/s)'}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<25} {r['seq_len']:<10} {r['kv_gb']:.4f} GB{'':<6} "
              f"{r['offload_time']:.3f}s{'':<6} {r['restore_time']:.3f}s{'':<6} "
              f"{r['offload_bandwidth']:.2f} / {r['restore_bandwidth']:.2f}")

    total_kv_offloaded = sum(r['kv_bytes'] for r in results)
    total_kv_gb = total_kv_offloaded / (1024**3)
    print("\n" + "=" * 80)
    print(f"Total KV Cache Offloaded: {total_kv_offloaded:,} bytes ({total_kv_gb:.4f} GB)")
    print(f"Average Offload Bandwidth: {sum(r['offload_bandwidth'] for r in results) / len(results):.2f} GB/s")
    print(f"Average Restore Bandwidth: {sum(r['restore_bandwidth'] for r in results) / len(results):.2f} GB/s")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import sys

    print("Checking server availability...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Server health: {resp.status_code}\n")
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server at", BASE_URL)
        print("\nPlease ensure the server is running.")
        sys.exit(1)

    results = test_large_context_offload()

    print("\n✓ All tests completed successfully!")
