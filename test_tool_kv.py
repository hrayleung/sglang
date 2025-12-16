#!/usr/bin/env python3
"""
Test script for TOOL_START/TOOL_END KV management API.

Usage:
1. Start the server in one terminal:
   python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000

2. Run this test in another terminal:
   python test_tool_kv.py
"""

import requests
import time
import json
import uuid

BASE_URL = "http://localhost:30000"


def test_basic_flow():
    """Test the basic TOOL_START/TOOL_END flow."""
    print("=" * 60)
    print("Testing Tool KV Management API")
    print("=" * 60)

    # 1. Open a session
    print("\n1. Opening session...")
    session_id = f"test-tool-session-{uuid.uuid4().hex[:8]}"
    resp = requests.post(
        f"{BASE_URL}/open_session",
        json={"capacity_of_str_len": 1024, "session_id": session_id},
    )
    print(f"   Response: {resp.status_code} - {resp.text}")
    
    if resp.status_code != 200:
        print("   Failed to open session. Is the server running?")
        return False

    returned_session_id = resp.json() if resp.text.startswith('"') else session_id
    session_id = returned_session_id
    print(f"   Session ID: {session_id}")

    # 2. Start a generation request with the session
    print("\n2. Starting generation with session...")
    gen_resp = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": "Hello, I need help with a calculation. What is 2 + 2?",
            "sampling_params": {
                "max_new_tokens": 50,
                "temperature": 0.7,
            },
            "session_params": {
                "id": session_id,
            },
            "stream": False,
        },
    )
    print(f"   Response: {gen_resp.status_code}")
    if gen_resp.status_code == 200:
        result = gen_resp.json()
        print(f"   Generated: {result.get('text', '')[:100]}...")

    # 3. Check KV metadata (before offload)
    print("\n3. Checking KV metadata (before offload)...")
    meta_resp = requests.get(f"{BASE_URL}/session/kv_meta?session_id={session_id}")
    print(f"   Response: {meta_resp.status_code} - {meta_resp.text}")

    # 4. Call TOOL_START to offload cached session KV to CPU
    print("\n4. Calling TOOL_START (offload KV to CPU)...")
    tool_start_resp = requests.post(
        f"{BASE_URL}/session/tool_start",
        json={"session_id": session_id, "mode": "cpu"},
    )
    print(f"   Response: {tool_start_resp.status_code}")
    tool_start_result = tool_start_resp.json()
    print(f"   Result: {json.dumps(tool_start_result, indent=2)}")

    if not tool_start_result.get("ok"):
        print(f"   TOOL_START failed: {tool_start_result.get('error')}")
        return False
    
    epoch = tool_start_result.get("epoch", 0)

    # 5. Simulate tool execution
    print("\n5. Simulating tool execution (sleeping 2 seconds)...")
    time.sleep(2)
    tool_result = "The calculation result is: 2 + 2 = 4"
    print(f"   Tool result: {tool_result}")

    # 6. Check KV metadata (during offload)
    print("\n6. Checking KV metadata (during offload)...")
    meta_resp = requests.get(f"{BASE_URL}/session/kv_meta?session_id={session_id}")
    print(f"   Response: {meta_resp.status_code} - {meta_resp.text}")

    # 7. Call TOOL_END to restore KV to GPU (best-effort)
    print("\n7. Calling TOOL_END (restore KV to GPU)...")
    tool_end_resp = requests.post(
        f"{BASE_URL}/session/tool_end",
        json={
            "session_id": session_id,
            "epoch": epoch,
            "tool_result": tool_result,
        },
    )
    print(f"   Response: {tool_end_resp.status_code}")
    tool_end_result = tool_end_resp.json()
    print(f"   Result: {json.dumps(tool_end_result, indent=2)}")
    if not tool_end_result.get("ok"):
        print(f"   TOOL_END failed: {tool_end_result.get('error')}")
        return False

    # 8. Check KV metadata (after restore)
    print("\n8. Checking KV metadata (after restore)...")
    meta_resp = requests.get(f"{BASE_URL}/session/kv_meta?session_id={session_id}")
    print(f"   Response: {meta_resp.status_code} - {meta_resp.text}")

    # 9. Close session
    print("\n9. Closing session...")
    close_resp = requests.post(
        f"{BASE_URL}/close_session",
        json={"session_id": session_id},
    )
    print(f"   Response: {close_resp.status_code}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    return True


def test_epoch_mismatch():
    """Test that epoch mismatch is properly rejected."""
    print("\n" + "=" * 60)
    print("Testing Epoch Mismatch Protection")
    print("=" * 60)

    session_id = f"test-epoch-session-{uuid.uuid4().hex[:8]}"
    
    # Open session
    resp = requests.post(
        f"{BASE_URL}/open_session",
        json={"capacity_of_str_len": 1024, "session_id": session_id},
    )
    if resp.status_code != 200:
        print(f"Failed to open session for epoch test: {resp.status_code} - {resp.text}")
        return

    # Create some cached KV for the session
    requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": "Quick test: what is 1+1?",
            "sampling_params": {"max_new_tokens": 32, "temperature": 0.7},
            "session_params": {"id": session_id},
            "stream": False,
        },
    )
    time.sleep(0.2)

    # Offload cached session KV
    tool_start_resp = requests.post(
        f"{BASE_URL}/session/tool_start",
        json={"session_id": session_id, "mode": "cpu"},
    )
    tool_start = tool_start_resp.json()
    if not tool_start.get("ok"):
        print(f"TOOL_START failed in epoch test: {json.dumps(tool_start, indent=2)}")
        requests.post(f"{BASE_URL}/close_session", json={"session_id": session_id})
        return
    epoch = tool_start.get("epoch", 0)

    # Try TOOL_END with wrong epoch (should fail)
    print("\nCalling TOOL_END with wrong epoch (should fail)...")
    resp = requests.post(
        f"{BASE_URL}/session/tool_end",
        json={"session_id": session_id, "epoch": epoch + 999, "tool_result": "test"},
    )
    result = resp.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    if not result.get("ok") and "Epoch mismatch" in (result.get("error") or ""):
        print("✓ Correctly rejected stale epoch!")
    elif not result.get("ok"):
        print("✓ Rejected (but not via epoch mismatch):", result.get("error"))
    else:
        print("✗ Should have rejected stale epoch")

    # Restore with correct epoch to clean up
    requests.post(
        f"{BASE_URL}/session/tool_end",
        json={"session_id": session_id, "epoch": epoch, "tool_result": "cleanup"},
    )

    # Cleanup
    requests.post(f"{BASE_URL}/close_session", json={"session_id": session_id})


def test_endpoints_exist():
    """Quick test that endpoints exist and respond."""
    print("\n" + "=" * 60)
    print("Testing Endpoint Availability")
    print("=" * 60)

    endpoints = [
        ("GET", "/session/kv_meta?session_id=nonexistent"),
        ("POST", "/session/tool_start"),
        ("POST", "/session/tool_end"),
    ]

    for method, path in endpoints:
        try:
            if method == "GET":
                resp = requests.get(f"{BASE_URL}{path}", timeout=5)
            else:
                payload = {"session_id": "test"}
                if path.endswith("/tool_start"):
                    payload["mode"] = "cpu"
                else:
                    payload["epoch"] = 0
                resp = requests.post(f"{BASE_URL}{path}", json=payload, timeout=5)
            print(f"  {method} {path}: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"  {method} {path}: Connection refused (server not running?)")
            return False
        except Exception as e:
            print(f"  {method} {path}: Error - {e}")

    return True


if __name__ == "__main__":
    import sys

    print("Checking server availability...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Server health: {resp.status_code}\n")
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server at", BASE_URL)
        print("\nPlease start the server first:")
        print("  python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000")
        sys.exit(1)

    # Run tests
    test_endpoints_exist()
    test_basic_flow()
    test_epoch_mismatch()
