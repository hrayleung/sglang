#!/usr/bin/env python3
"""
Test full KV offload and restore using the cached session path.

This tests the proper tool-calling semantic:
1. Create session
2. Complete a generation (KV gets cached in radix tree)
3. Call tool_start to offload cached KV to CPU
4. Call tool_end to restore KV back to GPU
5. Continue generation with restored KV (should reuse prefix)
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
        "capacity_of_str_len": 131072,  # Required field
    })
    print(f"Open session: {r.status_code}")
    if r.status_code != 200:
        print(f"   Error: {r.text[:200]}")
    return r.status_code == 200

def close_session(session_id: str):
    r = requests.post(f"{BASE_URL}/close_session", json={"session_id": session_id})
    print(f"Close session: {r.status_code}")

def generate(session_id: str, text: str, max_tokens: int = 50):
    r = requests.post(f"{BASE_URL}/generate", json={
        "text": text,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.7},
        "session_params": {"id": session_id},
    })
    if r.status_code == 200:
        return r.json()
    print(f"Generate failed: {r.status_code} {r.text}")
    return None

def tool_start(session_id: str):
    r = requests.post(f"{BASE_URL}/session/tool_start", json={
        "session_id": session_id,
        "mode": "cpu",
    })
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

def list_active():
    r = requests.get(f"{BASE_URL}/session/list_active_requests")
    return r.json()

def main():
    print("=" * 60)
    print("Full KV Offload/Restore Test (Cached Session Path)")
    print("=" * 60)
    
    if not check_server():
        print("Server not available!")
        return
    
    session_id = f"test-full-restore-{int(time.time())}"
    
    try:
        # Step 1: Create session
        print(f"\n1. Creating session: {session_id}")
        if not open_session(session_id):
            print("Failed to create session")
            return
        
        # Step 2: Complete a generation
        print("\n2. Running initial generation...")
        prompt = "The capital of France is"
        result = generate(session_id, prompt, max_tokens=30)
        if not result:
            print("Generation failed")
            return
        
        initial_output = result.get("text", "")
        print(f"   Prompt: {prompt}")
        print(f"   Output: {initial_output[:100]}...")
        
        # Wait for KV to be cached in radix tree
        time.sleep(0.5)
        
        # Debug: Check session state
        print("\n   Debug: Checking active requests...")
        active = list_active()
        print(f"   Active requests: {json.dumps(active, indent=2)}")
        
        # Step 3: Check KV meta before offload
        print("\n3. Checking KV meta before offload...")
        meta = get_kv_meta(session_id)
        print(f"   State: {meta.get('state')}")
        
        # Step 4: Offload KV to CPU
        print("\n4. Calling TOOL_START (offload to CPU)...")
        status, result = tool_start(session_id)
        print(f"   Status: {status}")
        print(f"   Result: {json.dumps(result, indent=2)}")
        
        if not result.get("ok"):
            print(f"   ✗ Offload failed: {result.get('error')}")
            return
        
        epoch = result.get("epoch")
        kv_bytes = result.get("kv_bytes", 0)
        print(f"   ✓ Offloaded! epoch={epoch}, kv_bytes={kv_bytes} ({kv_bytes/1024/1024:.2f} MB)")
        
        # Step 5: Check KV meta after offload
        print("\n5. Checking KV meta after offload...")
        meta = get_kv_meta(session_id)
        print(f"   State: {meta.get('state')}")
        print(f"   Tier: {meta.get('tier')}")
        print(f"   KV bytes: {meta.get('kv_bytes')}")
        
        # Step 6: Simulate tool execution
        print("\n6. Simulating tool execution (3 seconds)...")
        time.sleep(3)
        
        # Step 7: Restore KV to GPU
        print("\n7. Calling TOOL_END (restore to GPU)...")
        status, result = tool_end(session_id, epoch, tool_result="Tool executed successfully")
        print(f"   Status: {status}")
        print(f"   Result: {json.dumps(result, indent=2)}")
        
        if not result.get("ok"):
            print(f"   ✗ Restore failed: {result.get('error')}")
            return
        
        print(f"   ✓ Restored! state={result.get('state')}")
        
        # Step 8: Check KV meta after restore
        print("\n8. Checking KV meta after restore...")
        meta = get_kv_meta(session_id)
        print(f"   State: {meta.get('state')}")
        print(f"   Tier: {meta.get('tier')}")
        
        # Step 9: Continue generation (should reuse restored KV)
        print("\n9. Continuing generation (should reuse restored KV)...")
        continuation = " and it is known for"
        result = generate(session_id, prompt + initial_output + continuation, max_tokens=30)
        if result:
            print(f"   Continuation output: {result.get('text', '')[:100]}...")
            print("   ✓ Generation continued successfully!")
        else:
            print("   ✗ Continuation failed")
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
    finally:
        # Cleanup
        print(f"\nCleaning up session {session_id}...")
        close_session(session_id)

if __name__ == "__main__":
    main()
