#!/usr/bin/env python3
"""
Test script for TOOL_START/TOOL_END with streaming requests.

This test demonstrates the full flow:
1. Start a streaming generation
2. Call TOOL_START while it's running to offload KV
3. Call TOOL_END to restore and resume

Usage:
1. Start the server:
   python -m sglang.launch_server --model-path <model> --port 30000

2. Run this test:
   python test_tool_kv_streaming.py
"""

import requests
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://localhost:30000"


def test_list_active_requests():
    """Test the list_active_requests endpoint."""
    print("=" * 60)
    print("Testing List Active Requests")
    print("=" * 60)

    # Start a long generation in background
    print("\n1. Starting long generation in background...")
    
    generation_done = threading.Event()
    
    def long_generation():
        try:
            resp = requests.post(
                f"{BASE_URL}/generate",
                json={
                    "text": "Write a very long story about a robot learning to cook. Include many details.",
                    "sampling_params": {
                        "max_new_tokens": 200,
                        "temperature": 0.7,
                    },
                },
            )
        except Exception as e:
            print(f"   Generation error: {e}")
        finally:
            generation_done.set()

    gen_thread = threading.Thread(target=long_generation)
    gen_thread.start()
    
    # Wait a bit for generation to start
    time.sleep(0.5)
    
    # List active requests
    print("\n2. Listing active requests...")
    resp = requests.get(f"{BASE_URL}/session/list_active_requests")
    result = resp.json()
    print(f"   Response: {resp.status_code}")
    print(f"   Active requests: {json.dumps(result, indent=2)}")
    
    # Wait for generation to finish
    generation_done.wait(timeout=30)
    gen_thread.join(timeout=2)
    
    print("\n" + "=" * 60)
    print("List Active Requests test completed!")
    print("=" * 60)


def test_tool_start_with_wildcard():
    """Test TOOL_START with session_id='*' to target any running request."""
    print("=" * 60)
    print("Testing Tool Start with Wildcard Session ID")
    print("=" * 60)

    # Start a long generation in background
    print("\n1. Starting long generation in background...")
    
    generation_done = threading.Event()
    generation_result = {}
    
    def long_generation():
        try:
            resp = requests.post(
                f"{BASE_URL}/generate",
                json={
                    "text": "Write a very long story about a robot learning to cook. Include many details about ingredients, cooking techniques, and the robot's emotions. Make it at least 500 words.",
                    "sampling_params": {
                        "max_new_tokens": 300,
                        "temperature": 0.7,
                    },
                },
            )
            generation_result["response"] = resp.json()
        except Exception as e:
            generation_result["error"] = str(e)
        finally:
            generation_done.set()

    gen_thread = threading.Thread(target=long_generation)
    gen_thread.start()
    
    # Wait a bit for generation to start
    time.sleep(1)
    
    # List active requests first
    print("\n2. Listing active requests...")
    resp = requests.get(f"{BASE_URL}/session/list_active_requests")
    active_reqs = resp.json()
    print(f"   Active requests: {json.dumps(active_reqs, indent=2)}")
    
    # Try TOOL_START with wildcard
    print("\n3. Calling TOOL_START with session_id='*'...")
    tool_start_resp = requests.post(
        f"{BASE_URL}/session/tool_start",
        json={"session_id": "*", "mode": "cpu"},
    )
    result = tool_start_resp.json()
    print(f"   Response: {tool_start_resp.status_code}")
    print(f"   Result: {json.dumps(result, indent=2)}")

    if result.get("ok"):
        epoch = result.get("epoch", 0)
        kv_bytes = result.get('kv_bytes', 0)
        print(f"\n   ✓ Successfully offloaded! epoch={epoch}, kv_bytes={kv_bytes} ({kv_bytes/1024/1024:.2f} MB)")
        
        # Simulate tool execution for 30 seconds
        print("\n4. Simulating tool execution for 30 seconds...")
        print("   (Note: GPU memory in nvidia-smi may not drop because the KV cache is typically preallocated;")
        print("    check server logs for KV free-token changes and /session/kv_meta instead.)")
        for i in range(30):
            print(f"   Tool executing... {30-i}s remaining", end='\r')
            time.sleep(1)
        print("\n   Tool execution complete!")
        
        # Call TOOL_END
        print("\n5. Calling TOOL_END...")
        tool_end_resp = requests.post(
            f"{BASE_URL}/session/tool_end",
            json={
                "session_id": "*",
                "epoch": epoch,
                "tool_result": "The calculation shows: 42 is the answer.",
            },
        )
        end_result = tool_end_resp.json()
        print(f"   Response: {tool_end_resp.status_code}")
        print(f"   Result: {json.dumps(end_result, indent=2)}")
    else:
        print(f"\n   Note: {result.get('error')}")

    # Wait for generation to finish
    generation_done.wait(timeout=30)
    gen_thread.join(timeout=2)
    
    print("\n" + "=" * 60)
    print("Wildcard test completed!")
    print("=" * 60)


def test_tool_start_with_rid():
    """Test TOOL_START with specific request ID."""
    print("=" * 60)
    print("Testing Tool Start with Request ID")
    print("=" * 60)

    # Start a long generation in background
    print("\n1. Starting long generation in background...")
    
    generation_done = threading.Event()
    
    def long_generation():
        try:
            resp = requests.post(
                f"{BASE_URL}/generate",
                json={
                    "text": "Write a very long story about a robot learning to cook.",
                    "sampling_params": {
                        "max_new_tokens": 300,
                        "temperature": 0.7,
                    },
                },
            )
        except Exception as e:
            print(f"   Generation error: {e}")
        finally:
            generation_done.set()

    gen_thread = threading.Thread(target=long_generation)
    gen_thread.start()
    
    # Wait a bit for generation to start
    time.sleep(1)
    
    # List active requests to get the rid
    print("\n2. Listing active requests to get rid...")
    resp = requests.get(f"{BASE_URL}/session/list_active_requests")
    active_reqs = resp.json()
    print(f"   Active requests: {json.dumps(active_reqs, indent=2)}")
    
    if active_reqs.get("requests"):
        rid = active_reqs["requests"][0]["rid"]
        print(f"\n3. Found request with rid={rid}")
        
        # Try TOOL_START with specific rid
        print("\n4. Calling TOOL_START with target_rid...")
        tool_start_resp = requests.post(
            f"{BASE_URL}/session/tool_start",
            json={"session_id": "test-session", "mode": "cpu", "target_rid": rid},
        )
        result = tool_start_resp.json()
        print(f"   Response: {tool_start_resp.status_code}")
        print(f"   Result: {json.dumps(result, indent=2)}")

        # Call TOOL_END to release the CPU snapshot (no resume path for running-request offload).
        if result.get("ok"):
            epoch = result.get("epoch", 0)
            print("\n5. Calling TOOL_END (cleanup)...")
            tool_end_resp = requests.post(
                f"{BASE_URL}/session/tool_end",
                json={
                    "session_id": "test-session",
                    "epoch": epoch,
                    "tool_result": "cleanup",
                },
            )
            end_result = tool_end_resp.json()
            print(f"   Response: {tool_end_resp.status_code}")
            print(f"   Result: {json.dumps(end_result, indent=2)}")
    else:
        print("\n   No active requests found")

    # Wait for generation to finish
    generation_done.wait(timeout=30)
    gen_thread.join(timeout=2)
    
    print("\n" + "=" * 60)
    print("RID test completed!")
    print("=" * 60)


def test_simple_api_check():
    """Simple check that APIs respond correctly."""
    print("=" * 60)
    print("Simple API Check")
    print("=" * 60)

    # Test tool_start with non-existent session
    print("\n1. Testing tool_start with non-existent session...")
    resp = requests.post(
        f"{BASE_URL}/session/tool_start",
        json={"session_id": "does-not-exist", "mode": "cpu"},
    )
    result = resp.json()
    print(f"   ok={result.get('ok')}, error={result.get('error')}")
    assert result.get("ok") == False, "Should fail for non-existent session"
    print("   ✓ Correctly rejected")

    # Test tool_end with non-existent session
    print("\n2. Testing tool_end with non-existent session...")
    resp = requests.post(
        f"{BASE_URL}/session/tool_end",
        json={"session_id": "does-not-exist", "epoch": 0},
    )
    result = resp.json()
    print(f"   ok={result.get('ok')}, error={result.get('error')}")
    assert result.get("ok") == False, "Should fail for non-existent session"
    print("   ✓ Correctly rejected")

    # Test kv_meta
    print("\n3. Testing kv_meta...")
    resp = requests.get(f"{BASE_URL}/session/kv_meta?session_id=test")
    result = resp.json()
    print(f"   state={result.get('state')}")
    print("   ✓ API responds")

    # Test list_active_requests
    print("\n4. Testing list_active_requests...")
    resp = requests.get(f"{BASE_URL}/session/list_active_requests")
    result = resp.json()
    print(f"   requests={result.get('requests')}")
    print("   ✓ API responds")

    print("\n" + "=" * 60)
    print("All API checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    print("Checking server availability...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Server health: {resp.status_code}\n")
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server at", BASE_URL)
        sys.exit(1)

    test_simple_api_check()
    print("\n")
    test_list_active_requests()
    print("\n")
    test_tool_start_with_wildcard()
    print("\n")
    test_tool_start_with_rid()
