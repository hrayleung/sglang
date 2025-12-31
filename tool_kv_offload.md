# Tool-calling KV offload/restore (SGLang modifications)

## Scope
Documentation of the locally modified KV cache offload/restore flow driven by tool-calling semantics (TOOL_START/TOOL_END), covering both shadow-copy (V1) and HiRadixCache-native (V2) implementations.

## Components & entry points
- HTTP endpoints: `/session/tool_start`, `/session/tool_end`, `/session/kv_meta` (handled in `python/sglang/srt/entrypoints/http_server.py`).
- Scheduler integration: lazy manager selection in `python/sglang/srt/managers/scheduler.py:2682-2707`.
  - Uses V2 (HiRadixCache) when `tree_cache` exposes `offload_for_tool` & `preload_for_tool`.
  - Falls back to V1 (shadow copy) otherwise.
- Managers:
  - V1: `ToolKVManager` (`python/sglang/srt/managers/tool_kv_manager.py:155-994`).
  - V2: `ToolKVManagerV2` (`python/sglang/srt/managers/tool_kv_manager.py:995-1422`).
- HiRadixCache primitives: `offload_for_tool`, `preload_for_tool`, `load_back`, lock helpers (`python/sglang/srt/mem_cache/hiradix_cache.py:373-482, 521-578`).
- Radix cache locking/leaf collection: `inc_lock_ref/dec_lock_ref`, `_collect_leaves` (`python/sglang/srt/mem_cache/radix_cache.py:571-602, 751-763`).

## What changed locally vs upstream
- Added tool-aware KV control plane (tool_start/tool_end) with semantic pause/resume.
- V1 (shadow copy) hardened: bigram key handling, null CPU copy checks, length validation, write-ack verification, try/finally cleanup, branch deletability checks.
- V2 added: native HiRadixCache integration (commit `b55a9ccdf`), using host protection and TTL cleanup to avoid leaks.
- Scheduler auto-selects V2 when hierarchical cache enabled; V1 is fallback.

## Strategy overview
### V2 (HiRadixCache-native, preferred)
- Offload (`ToolKVManagerV2.tool_start`):
  1) Find session’s last radix node (`_find_session_node`).
  2) `HiRadixCache.offload_for_tool(node, protect=True)`:
     - GPU→CPU write (`write_backup` + `writing_check`).
     - Evict GPU value (`_evict_backuped` → `node.value=None`).
     - Protect CPU buffer (`node.protect_host()`).
  3) Track session→node, timestamp, meta (state=OFFLOADED_TO_CPU, tier=CPU, epoch++).
- Restore (`ToolKVManagerV2.tool_end`):
  1) Validate epoch/meta.
  2) `HiRadixCache.preload_for_tool(node, release=True)`:
     - CPU→GPU (`load_back`), wait via `loading_check`.
     - Release host protection.
  3) Clear tracking; meta→RUNNABLE, tier=GPU.
- TTL cleanup: `cleanup_stale_sessions()` releases host locks and invalidates meta after 1h (`tool_kv_manager.py:1322-1359`).

### V1 (shadow copy, fallback)
- Offload cached/finished sessions: build radix key, copy leaf-branch KV to CPU (`kv_cache.get_cpu_copy`), prune branch from radix tree; store `_kv_cache_cpu`, radix tokens, lengths, epoch (`tool_kv_manager.py:308-399`).
- Restore: reinsert CPU snapshot when tokens+metadata exist; otherwise just mark runnable and drop CPU buffers (`tool_kv_manager.py:850-929`).

## Data that is offloaded
- V2: The radix node’s KV payload (`node.value`) is written to CPU (`node.host_value`); GPU value set to `None`; node stays in-tree, host buffer protected.
- V1: CPU snapshot of the leaf branch (`_kv_cache_cpu` or `_k_host/_v_host/_kv_host`), plus radix path tokens and extra key metadata for reinsertion.

## Known caveats / potential bugs to watch
1) Host lock leak on preload failure: `preload_for_tool` returns early if `load_back` returns None, without releasing `host_ref_counter` acquired during offload (`hiradix_cache.py:440-482`).
2) `ongoing_load_back` concurrency: set/pop without explicit locking between `load_back` and `loading_check` (`hiradix_cache.py:303-316, 521-578`).
3) Stale host buffers on GPU eviction: `_evict_backuped` frees GPU but leaves `host_value`; if the node is later removed without host eviction, CPU memory can leak (`hiradix_cache.py:365-371`).
4) Leaf collection ignores host locks: `_collect_leaves` filters only `lock_ref==0`, not `host_ref_counter`, so `evict_host` heaps include nodes that will be skipped later (`radix_cache.py:751-763`).
5) Session→node cache validity: `_find_session_node` only checks `hasattr(node, "id")`; stale nodes (removed/GC’d) may remain (`tool_kv_manager.py:1056-1096`).
6) TTL cleanup requires caller: `cleanup_stale_sessions()` is defined but not auto-invoked; needs periodic scheduler call to release host locks if tool_end never arrives (`tool_kv_manager.py:1322-1359`).
7) Offload/evict race window: during `offload_for_tool`, GPU value goes to None after backup; concurrent eviction could see transient states despite `lock_ref` protection (`hiradix_cache.py:373-438`).

## Operational guidance
- Enable hierarchical cache (`--enable-hierarchical-cache`) to use V2; otherwise V1 shadow copy runs.
- Orchestrator must call `tool_end` with the returned epoch; otherwise rely on TTL cleanup (V2) to release host protections.
- On V2 preload failure (OOM/threshold), the session stays offloaded and host-locked; manual cleanup or TTL required.
- Epoch increments on each offload; callers must echo epoch on restore.
- V2 keeps radix sharing intact: nodes stay in-tree; only payload tiers change.

## Quick references
- Offload path (V2): `scheduler.py:2682-2722` → `tool_kv_manager.py:1120-1216` → `hiradix_cache.py:373-438`.
- Restore path (V2): `scheduler.py:2731-2766` → `tool_kv_manager.py:1227-1311` → `hiradix_cache.py:440-482`.
- TTL cleanup (V2): `tool_kv_manager.py:1322-1359`.
- Shadow copy paths (V1): offload `tool_kv_manager.py:308-399`; restore `tool_kv_manager.py:850-929`.
