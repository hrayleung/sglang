# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Semantic-driven KV cache management for tool calling.

Provides TOOL_START/TOOL_END control plane for explicit KV offload/restore
when an external agent/orchestrator needs to pause generation during tool execution.

MVP Implementation:
- tool_start: Backup KV to CPU, then abort the request (proper memory release)
- tool_end: Restore cached-session KV back to GPU (best-effort) and mark runnable
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SessionKVState(Enum):
    """State of a session's KV cache."""
    RUNNABLE = "RUNNABLE"
    OFFLOADED_TO_CPU = "OFFLOADED_TO_CPU"
    INVALID = "INVALID"


class ToolKVError(Exception):
    """Base exception for Tool KV management errors."""
    pass


class SessionNotFoundError(ToolKVError):
    """Session not found."""
    pass


class SessionStateError(ToolKVError):
    """Invalid session state for the requested operation."""
    pass


class EpochMismatchError(ToolKVError):
    """Epoch mismatch - stale request."""
    pass


@dataclass
class SessionKVMeta:
    """Metadata for a session's KV cache state."""
    session_id: str
    state: SessionKVState = SessionKVState.RUNNABLE
    epoch: int = 0
    tier: str = "GPU"  # "GPU", "CPU"
    kv_bytes: int = 0
    seq_len: int = 0
    last_access: float = field(default_factory=time.time)
    
    # For tracking the request
    rid: Optional[str] = None
    
    # Stored KV data (CPU tensors)
    _k_host: Optional[torch.Tensor] = None
    _v_host: Optional[torch.Tensor] = None
    _kv_host: Optional[torch.Tensor] = None  # For MLA style

    # Stored KV data (CPU copy via KVCache.get_cpu_copy)
    _kv_cache_cpu: Optional[Any] = None

    # Cached radix key tokens (internal representation, page-aligned)
    _radix_key_tokens: Optional[List[int]] = None
    _radix_extra_key: Optional[str] = None
    _radix_key_is_bigram: bool = False
    _radix_offloaded_len: int = 0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "epoch": self.epoch,
            "tier": self.tier,
            "kv_bytes": self.kv_bytes,
            "seq_len": self.seq_len,
            "last_access": self.last_access,
            "rid": self.rid,
        }


@dataclass 
class ToolStartResult:
    """Result of tool_start operation."""
    ok: bool
    session_id: str
    state: str
    kv_bytes: int = 0
    epoch: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {
            "ok": self.ok,
            "session_id": self.session_id,
            "state": self.state,
            "kv_bytes": self.kv_bytes,
            "epoch": self.epoch,
        }
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class ToolEndResult:
    """Result of tool_end operation."""
    ok: bool
    session_id: str
    state: str
    epoch: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {
            "ok": self.ok,
            "session_id": self.session_id,
            "state": self.state,
            "epoch": self.epoch,
        }
        if self.error:
            result["error"] = self.error
        return result


class ToolKVManager:
    """
    Manager for semantic-driven KV cache offload/restore operations.
    
    MVP Implementation:
    - tool_start: Copy KV to CPU pinned memory, then let the request finish/abort normally
    - tool_end: Restores cached-session KV when available (radix-cache path)
    
    The key insight is that we backup the KV state before the request is aborted,
    so we have a snapshot that could be restored later (future enhancement).
    """
    
    def __init__(self, scheduler: "Scheduler"):
        self.scheduler = scheduler
        self.lock = threading.RLock()
        
        # Session KV metadata: session_id -> SessionKVMeta
        self.session_kv_meta: Dict[str, SessionKVMeta] = {}
        
        logger.info("ToolKVManager initialized. HiCache enabled: False")
    
    def _get_or_create_meta(self, session_id: str) -> SessionKVMeta:
        """Get or create session KV metadata."""
        if session_id not in self.session_kv_meta:
            self.session_kv_meta[session_id] = SessionKVMeta(session_id=session_id)
        return self.session_kv_meta[session_id]

    @staticmethod
    def _sum_tensor_bytes(obj: Any) -> int:
        """Recursively sum tensor nbytes for nested structures."""
        if obj is None:
            return 0
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(ToolKVManager._sum_tensor_bytes(x) for x in obj)
        if isinstance(obj, dict):
            return sum(ToolKVManager._sum_tensor_bytes(v) for v in obj.values())
        return 0

    def _get_session(self, session_id: str):
        return getattr(self.scheduler, "sessions", {}).get(session_id)

    def _find_session_req(self, session_id: str, rid: Optional[str] = None) -> Optional["Req"]:
        """Find the best candidate request in a session (finished or not)."""
        session = self._get_session(session_id)
        if session is None:
            return None

        req_nodes = getattr(session, "req_nodes", None)
        if not req_nodes:
            return None

        if rid is not None:
            node = req_nodes.get(rid)
            return getattr(node, "req", None) if node is not None else None

        # Choose a leaf request in the session tree (most recent by timestamps).
        candidates = []
        for node in req_nodes.values():
            req = getattr(node, "req", None)
            if req is None:
                continue
            childs = getattr(node, "childs", None)
            if childs is not None and len(childs) != 0:
                continue
            candidates.append(req)

        if not candidates:
            # Fall back to the last inserted request.
            try:
                last_node = next(reversed(req_nodes.values()))
                return getattr(last_node, "req", None)
            except Exception:
                return None

        def sort_key(req: "Req") -> Tuple[float, str]:
            ts = getattr(req, "time_stats", None)
            t = 0.0
            if ts is not None:
                t = max(
                    getattr(ts, "completion_time", 0.0),
                    getattr(ts, "forward_entry_time", 0.0),
                    getattr(ts, "wait_queue_entry_time", 0.0),
                    getattr(ts, "lb_entry_time", 0.0),
                )
            # Fall back to last_tic (monotonic, local) and rid.
            t = max(t, float(getattr(req, "last_tic", 0.0)))
            return (t, getattr(req, "rid", ""))

        return max(candidates, key=sort_key)

    def _get_tree_cache(self):
        return getattr(self.scheduler, "tree_cache", None)

    def _get_radix_root(self, tree_cache) -> Optional[TreeNode]:
        return getattr(tree_cache, "root_node", None)

    def _radix_path_tokens(self, leaf: TreeNode, root: TreeNode) -> List[int]:
        segments: List[List[int]] = []
        node = leaf
        while node is not None and node != root:
            segments.append(getattr(getattr(node, "key", None), "token_ids", []) or [])
            node = getattr(node, "parent", None)
        tokens: List[int] = []
        for seg in reversed(segments):
            tokens.extend(seg)
        return tokens

    def _plan_radix_branch_delete(self, leaf: TreeNode, root: TreeNode) -> List[TreeNode]:
        """
        Collect a deletable radix branch, starting from a leaf and walking upward
        while the branch remains exclusive (parent has only this child) and unlocked.
        """
        nodes: List[TreeNode] = []
        node = leaf
        while node is not None and node != root:
            if getattr(node, "lock_ref", 0) != 0:
                break
            if getattr(node, "children", None) and len(node.children) != 0:
                break
            nodes.append(node)

            parent = getattr(node, "parent", None)
            if parent is None or parent == root:
                break
            if getattr(parent, "lock_ref", 0) != 0:
                break
            # Only continue if parent will become a leaf after deleting this node.
            if getattr(parent, "children", None) is None or len(parent.children) != 1:
                break

            node = parent
        return nodes

    def _delete_radix_leaf(self, tree_cache, node: TreeNode) -> None:
        # Prefer the cache's own helper for bookkeeping.
        if hasattr(tree_cache, "_delete_leaf"):
            tree_cache._delete_leaf(node)
        else:
            child_key_fn = getattr(tree_cache, "get_child_key_fn", None)
            if child_key_fn is None:
                raise RuntimeError("tree_cache does not support leaf deletion")
            key = child_key_fn(node.key)
            v = node.parent.children.pop(key, None)
            assert v == node, f"parent does not have child key, {key}"
            # Best-effort bookkeeping
            if hasattr(tree_cache, "evictable_size_"):
                tree_cache.evictable_size_ -= len(node.key)

        if hasattr(tree_cache, "_record_remove_event"):
            tree_cache._record_remove_event(node)

    def _offload_cached_session_kv(self, session_id: str, rid: Optional[str]) -> ToolStartResult:
        """
        Offload cached KV for a *finished/idle* session by pruning its radix-cache branch.

        This is the typical tool-calling semantic: generation already ended, but the
        session's KV is kept in the radix cache for reuse. We snapshot the KV to CPU,
        then free the GPU cache lines by removing the leaf branch from the radix tree.
        """
        tree_cache = self._get_tree_cache()
        root = self._get_radix_root(tree_cache)
        if tree_cache is None or root is None:
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error="Prefix cache (tree_cache) is not available; cannot offload session KV",
            )

        req = self._find_session_req(session_id, rid=rid)
        if req is None:
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error=f"Session {session_id} not found or has no requests",
            )

        # Build a radix key for this request's full token history (radix cache will page-align).
        token_ids = (req.origin_input_ids or []) + (req.output_ids or [])
        if not token_ids:
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error="Session request has empty token history; nothing to offload",
            )

        extra_key = getattr(req, "extra_key", None)
        is_bigram = bool(getattr(tree_cache, "is_eagle", False))
        # Use the detected is_bigram value for consistency
        radix_key = RadixKey(token_ids=token_ids, extra_key=extra_key, is_bigram=is_bigram)
        match = tree_cache.match_prefix(radix_key)
        device_indices = match.device_indices
        leaf = match.last_device_node

        if device_indices is None or len(device_indices) == 0:
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error="No cached KV found on GPU for this session (maybe finished insertion disabled)",
            )

        if leaf is None or leaf == root:
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error="Cannot locate cached leaf node for this session in radix cache",
            )

        # Offload only if this key ends at a leaf branch (safe to prune).
        if getattr(leaf, "children", None) and len(leaf.children) != 0:
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error="Cached KV is shared (node has children); refusing to offload to avoid breaking other cached prefixes",
            )

        nodes_to_delete = self._plan_radix_branch_delete(leaf, root)
        if not nodes_to_delete:
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error="Cached KV branch is locked or not deletable; cannot offload now",
            )

        # Determine the full matched key tokens (internal representation) for later restore.
        key_tokens = self._radix_path_tokens(leaf, root)
        # Determine the pruned suffix length.
        pruned_len = sum(len(n.key) for n in nodes_to_delete)

        # Assemble indices to copy to CPU for the pruned branch (root->leaf order).
        value_segments = [n.value for n in reversed(nodes_to_delete)]
        if any(v is None for v in value_segments):
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error="Unexpected evicted node encountered; hierarchical cache offload is not supported in ToolKVManager yet",
            )
        pruned_indices = torch.cat(value_segments) if len(value_segments) > 1 else value_segments[0]

        available_before = self.scheduler.token_to_kv_pool_allocator.available_size()

        # Backup KV to CPU.
        # NOTE: PagedTokenToKVPoolAllocator may return None (not implemented).
        kv_cache_cpu = self.scheduler.token_to_kv_pool_allocator.get_cpu_copy(pruned_indices)
        if kv_cache_cpu is None:
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error="KV cache CPU copy not supported by current allocator (paged allocator?)",
            )
        kv_bytes = self._sum_tensor_bytes(kv_cache_cpu)
        if kv_bytes == 0:
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error="KV cache backup resulted in 0 bytes; backup may have failed",
            )

        # Free GPU KV by pruning the radix tree branch (leaf->root order).
        # NOTE: This assumes single-threaded scheduler access. If concurrent eviction
        # is possible, nodes should be locked before backup.
        for node in nodes_to_delete:
            self.scheduler.token_to_kv_pool_allocator.free(node.value)
            self._delete_radix_leaf(tree_cache, node)

        available_after = self.scheduler.token_to_kv_pool_allocator.available_size()

        meta = self._get_or_create_meta(session_id)
        meta.state = SessionKVState.OFFLOADED_TO_CPU
        meta.tier = "CPU"
        meta.epoch += 1
        meta.last_access = time.time()
        meta.seq_len = len(key_tokens)
        meta.kv_bytes = kv_bytes
        meta.rid = getattr(req, "rid", None)
        meta._kv_cache_cpu = kv_cache_cpu
        meta._radix_key_tokens = key_tokens
        meta._radix_extra_key = extra_key
        meta._radix_key_is_bigram = is_bigram
        meta._radix_offloaded_len = int(pruned_len)

        logger.info(
            f"Offloaded cached session {session_id} to CPU: rid={meta.rid}, "
            f"cached_len={len(key_tokens)}, offloaded_len={pruned_len}, kv_bytes={kv_bytes}, "
            f"kv_free_tokens={available_before}->{available_after}"
        )

        return ToolStartResult(
            ok=True,
            session_id=session_id,
            state=meta.state.value,
            kv_bytes=kv_bytes,
            epoch=meta.epoch,
        )

    def _restore_cached_session_kv(self, meta: SessionKVMeta) -> Optional[str]:
        """Restore a previously offloaded session KV back into the radix cache."""
        tree_cache = self._get_tree_cache()
        root = self._get_radix_root(tree_cache)
        if tree_cache is None or root is None:
            return "Prefix cache (tree_cache) is not available; cannot restore session KV"

        if not meta._radix_key_tokens or meta._kv_cache_cpu is None:
            return "No cached CPU KV backup found for this session"

        radix_key = RadixKey(
            token_ids=meta._radix_key_tokens,
            extra_key=meta._radix_extra_key,
            is_bigram=meta._radix_key_is_bigram,
        )

        match = tree_cache.match_prefix(radix_key)
        existing = match.device_indices
        prefix_len = int(len(existing))
        total_len = int(len(meta._radix_key_tokens))
        missing_len = total_len - prefix_len

        if missing_len != int(meta._radix_offloaded_len):
            return (
                f"Cannot restore: cached prefix changed (missing_len={missing_len}, "
                f"expected_offloaded_len={meta._radix_offloaded_len})"
            )

        if missing_len == 0:
            # Nothing to restore.
            return None

        new_indices = self.scheduler.token_to_kv_pool_allocator.alloc(missing_len)
        if new_indices is None:
            # Best-effort: evict some cache entries to make room, then retry.
            if hasattr(tree_cache, "evict"):
                tree_cache.evict(missing_len)
            new_indices = self.scheduler.token_to_kv_pool_allocator.alloc(missing_len)
        if new_indices is None:
            return f"Not enough free KV cache to restore {missing_len} tokens"

        # Validate allocated length matches expected
        if len(new_indices) != missing_len:
            self.scheduler.token_to_kv_pool_allocator.free(new_indices)
            return f"Allocated indices length mismatch ({len(new_indices)} != {missing_len})"

        # Use try/finally to ensure cleanup on failure
        try:
            self.scheduler.token_to_kv_pool_allocator.load_cpu_copy(meta._kv_cache_cpu, new_indices)

            full_indices = (
                torch.cat([existing, new_indices]) if prefix_len > 0 else new_indices
            )
            if len(full_indices) != total_len:
                raise RuntimeError(
                    f"Internal error: restored indices length mismatch ({len(full_indices)} != {total_len})"
                )

            if not hasattr(tree_cache, "insert"):
                raise RuntimeError("Prefix cache does not support insert(); cannot restore")

            # Check insert return value - it returns number of tokens inserted
            inserted = tree_cache.insert(radix_key, full_indices)
            if inserted == 0 and missing_len > 0:
                logger.warning(
                    f"RadixCache insert returned 0 for session {meta.session_id}, "
                    f"expected to insert {missing_len} tokens. Key may already exist."
                )

        except Exception as e:
            # Cleanup: free allocated indices on failure
            logger.error(f"Failed to restore KV for session {meta.session_id}: {e}")
            self.scheduler.token_to_kv_pool_allocator.free(new_indices)
            return f"Failed to restore KV: {str(e)}"

        # Clear CPU backup to release memory.
        meta._kv_cache_cpu = None
        meta.tier = "GPU"
        meta.last_access = time.time()

        logger.info(
            f"Restored cached session {meta.session_id} to GPU: rid={meta.rid}, restored_len={missing_len}"
        )
        return None
    
    def _find_request_by_session(self, session_id: str) -> Optional["Req"]:
        """Find a request associated with a session."""
        if self.scheduler.running_batch:
            for req in self.scheduler.running_batch.reqs:
                if getattr(req, 'session_id', None) == session_id:
                    return req
        
        for req in self.scheduler.waiting_queue:
            if getattr(req, 'session_id', None) == session_id:
                return req
        
        if self.scheduler.chunked_req:
            if getattr(self.scheduler.chunked_req, 'session_id', None) == session_id:
                return self.scheduler.chunked_req
        
        return None
    
    def _find_request_by_rid(self, rid: str) -> Optional["Req"]:
        """Find a request by its request ID."""
        if self.scheduler.running_batch:
            for req in self.scheduler.running_batch.reqs:
                if req.rid == rid:
                    return req
        
        for req in self.scheduler.waiting_queue:
            if req.rid == rid:
                return req
        
        if self.scheduler.chunked_req:
            if self.scheduler.chunked_req.rid == rid:
                return self.scheduler.chunked_req
        
        return None
    
    def _get_any_running_request(self) -> Optional["Req"]:
        """Get any running request (for testing/debugging)."""
        if self.scheduler.running_batch and self.scheduler.running_batch.reqs:
            return self.scheduler.running_batch.reqs[0]
        
        if self.scheduler.chunked_req:
            return self.scheduler.chunked_req
        
        if self.scheduler.waiting_queue:
            return self.scheduler.waiting_queue[0]
        
        return None
    
    def list_active_requests(self) -> list:
        """List all active requests (for debugging)."""
        requests = []
        
        if self.scheduler.running_batch:
            for req in self.scheduler.running_batch.reqs:
                requests.append({
                    "rid": req.rid,
                    "session_id": getattr(req, 'session_id', None),
                    "location": "running_batch",
                    "seq_len": len(req.origin_input_ids) + len(req.output_ids),
                    "output_len": len(req.output_ids),
                })
        
        if self.scheduler.chunked_req:
            req = self.scheduler.chunked_req
            requests.append({
                "rid": req.rid,
                "session_id": getattr(req, 'session_id', None),
                "location": "chunked_req",
                "seq_len": len(req.origin_input_ids) + len(req.output_ids),
                "output_len": len(req.output_ids),
            })
        
        for req in self.scheduler.waiting_queue:
            requests.append({
                "rid": req.rid,
                "session_id": getattr(req, 'session_id', None),
                "location": "waiting_queue",
                "seq_len": len(req.origin_input_ids) + len(req.output_ids),
                "output_len": len(req.output_ids),
            })
        
        return requests
    
    def get_kv_meta(self, session_id: str) -> Optional[SessionKVMeta]:
        """Get KV metadata for a session."""
        with self.lock:
            return self.session_kv_meta.get(session_id)
    
    def _abort_request(self, req: "Req"):
        """
        Abort a request to free its GPU memory.
        
        This marks the request as finished with an abort reason,
        which will cause the scheduler to release its KV cache.
        """
        from sglang.srt.managers.schedule_batch import FINISH_ABORT
        
        # Mark the request as aborted
        req.to_finish = FINISH_ABORT(
            message="Request paused for tool execution (KV backed up to CPU)"
        )
        
        logger.info(f"Marked request {req.rid} for abort (KV backed up to CPU)")
    
    def _backup_kv_to_cpu(self, req: "Req", meta: SessionKVMeta) -> bool:
        """
        Backup KV cache from GPU to CPU pinned memory.
        
        This creates a snapshot of the KV state that could be used for restoration.
        """
        try:
            req_pool_idx = req.req_pool_idx
            if req_pool_idx < 0:
                logger.warning(f"Request {req.rid} has no allocated KV cache")
                return False
            
            # Calculate sequence length
            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            if seq_len == 0:
                logger.warning(f"Request {req.rid} has zero sequence length")
                return False
            
            # Get the token indices
            token_indices = self.scheduler.req_to_token_pool.req_to_token[req_pool_idx, :seq_len]
            
            # Get KV pool
            kv_pool = self.scheduler.token_to_kv_pool_allocator.get_kvcache()
            
            # Copy KV to CPU pinned memory
            if hasattr(kv_pool, 'k_buffer') and hasattr(kv_pool, 'v_buffer'):
                # MHA style
                num_layers = len(kv_pool.k_buffer)
                k_shape = kv_pool.k_buffer[0].shape
                v_shape = kv_pool.v_buffer[0].shape
                
                k_host = torch.empty(
                    (num_layers, seq_len) + k_shape[1:],
                    dtype=kv_pool.k_buffer[0].dtype,
                    device='cpu',
                    pin_memory=True
                )
                v_host = torch.empty(
                    (num_layers, seq_len) + v_shape[1:],
                    dtype=kv_pool.v_buffer[0].dtype,
                    device='cpu',
                    pin_memory=True
                )
                
                for layer_idx in range(num_layers):
                    k_host[layer_idx].copy_(kv_pool.k_buffer[layer_idx][token_indices], non_blocking=True)
                    v_host[layer_idx].copy_(kv_pool.v_buffer[layer_idx][token_indices], non_blocking=True)
                
                torch.cuda.synchronize()
                
                meta._k_host = k_host
                meta._v_host = v_host
                meta.kv_bytes = k_host.numel() * k_host.element_size() + v_host.numel() * v_host.element_size()
                
            elif hasattr(kv_pool, 'kv_buffer'):
                # MLA style
                num_layers = len(kv_pool.kv_buffer)
                kv_shape = kv_pool.kv_buffer[0].shape
                
                kv_host = torch.empty(
                    (num_layers, seq_len) + kv_shape[1:],
                    dtype=kv_pool.kv_buffer[0].dtype,
                    device='cpu',
                    pin_memory=True
                )
                
                for layer_idx in range(num_layers):
                    kv_host[layer_idx].copy_(kv_pool.kv_buffer[layer_idx][token_indices], non_blocking=True)
                
                torch.cuda.synchronize()
                
                meta._kv_host = kv_host
                meta.kv_bytes = kv_host.numel() * kv_host.element_size()
            else:
                logger.error("Unknown KV pool type")
                return False
            
            meta.seq_len = seq_len
            meta.rid = req.rid
            
            # Log CPU memory info
            import psutil
            cpu_mem = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Backed up KV for request {req.rid}: seq_len={seq_len}, kv_bytes={meta.kv_bytes} ({meta.kv_bytes/1024/1024:.2f} MB), process RSS: {cpu_mem:.2f} MB")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to backup KV for request {req.rid}: {e}")
            return False
    
    def tool_start(
        self,
        session_id: str,
        mode: str = "cpu",
        rid: Optional[str] = None,
    ) -> ToolStartResult:
        """
        Handle TOOL_START request: backup KV to CPU.
        
        Two supported paths:
        - Running-request path (debug): snapshot KV then abort the request.
        - Cached-session path (tool-call semantic): snapshot KV from the radix cache, prune the branch to free KV slots.
        
        Args:
            session_id: Session to pause (use "*" for any running request)
            mode: Offload mode ("cpu" only for now)
            rid: Optional specific request ID
        """
        try:
            if mode != "cpu":
                return ToolStartResult(
                    ok=False,
                    session_id=session_id,
                    state="ERROR",
                    error=f"Unsupported mode: {mode}. Only 'cpu' is supported."
                )

            with self.lock:
                # 1) Try active request offload (mid-generation).
                req = None
                if rid:
                    req = self._find_request_by_rid(rid)
                elif session_id == "*":
                    req = self._get_any_running_request()
                    if req:
                        logger.info(f"Using any running request: rid={req.rid}")
                else:
                    req = self._find_request_by_session(session_id)

                if req is not None:
                    meta = self._get_or_create_meta(session_id)

                    if self._backup_kv_to_cpu(req, meta):
                        meta.state = SessionKVState.OFFLOADED_TO_CPU
                        meta.tier = "CPU"
                        meta.epoch += 1
                        meta.last_access = time.time()

                        # Abort the request to free KV slots (note: GPU memory pool stays allocated).
                        self._abort_request(req)

                        logger.info(
                            f"Offloaded running request for session {session_id} to CPU: "
                            f"seq_len={meta.seq_len}, kv_bytes={meta.kv_bytes}, epoch={meta.epoch}"
                        )
                        return ToolStartResult(
                            ok=True,
                            session_id=session_id,
                            state=meta.state.value,
                            kv_bytes=meta.kv_bytes,
                            epoch=meta.epoch,
                        )

                    return ToolStartResult(
                        ok=False,
                        session_id=session_id,
                        state="ERROR",
                        error="Failed to backup KV to CPU for running request",
                    )

                # 2) If no active request found, try cached session offload (tool-call semantic).
                if session_id != "*":
                    return self._offload_cached_session_kv(session_id=session_id, rid=rid)

                active_reqs = self.list_active_requests()
                logger.warning(
                    f"No request found for session {session_id}. Active requests: {active_reqs}"
                )
                return ToolStartResult(
                    ok=False,
                    session_id=session_id,
                    state="ERROR",
                    error=(
                        f"No active request found for session {session_id}. "
                        f"Active requests: {len(active_reqs)}. "
                        f"Use session_id='*' to target any running request."
                    ),
                )
        
        except Exception as e:
            logger.exception(f"Unexpected error in tool_start for session {session_id}")
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error=f"Internal error: {str(e)}",
            )
    
    def tool_end(
        self,
        session_id: str,
        epoch: int,
        tool_result: Optional[str] = None,
    ) -> ToolEndResult:
        """
        Handle TOOL_END request: mark session as ready.
        
        MVP: We just mark the session as ready. The actual KV restore
        would need to be integrated with the next generation request.
        
        Args:
            session_id: Session to resume
            epoch: Expected epoch (must match)
            tool_result: Optional tool output (stored for future use)
        """
        try:
            with self.lock:
                if session_id not in self.session_kv_meta:
                    return ToolEndResult(
                        ok=False,
                        session_id=session_id,
                        state="ERROR",
                        error=f"Session {session_id} not found"
                    )
                
                meta = self.session_kv_meta[session_id]
                
                # Validate epoch
                if meta.epoch != epoch:
                    return ToolEndResult(
                        ok=False,
                        session_id=session_id,
                        state="ERROR",
                        error=f"Epoch mismatch: expected {epoch}, got {meta.epoch}"
                    )
                
                # Validate state
                if meta.state != SessionKVState.OFFLOADED_TO_CPU:
                    return ToolEndResult(
                        ok=False,
                        session_id=session_id,
                        state="ERROR",
                        error=f"Cannot restore session in state {meta.state.value}"
                    )
                
                # Store tool result for future use
                if tool_result:
                    meta._pending_tool_result = tool_result
                    logger.info(f"Stored tool result for session {session_id}: {len(tool_result)} chars")

                # Restore cached session KV back to GPU when we have a radix-cache snapshot.
                # For the "running request abort" path (debug/testing), we may only have
                # a raw KV tensor snapshot without an integrated restore path.
                if meta._kv_cache_cpu is not None and meta._radix_key_tokens:
                    restore_error = self._restore_cached_session_kv(meta)
                    if restore_error is not None:
                        return ToolEndResult(
                            ok=False,
                            session_id=session_id,
                            state="ERROR",
                            epoch=meta.epoch,
                            error=restore_error,
                        )
                    restored = True
                else:
                    logger.info(
                        f"Session {session_id} marked as ready (KV backup remains on CPU; no restore path)"
                    )
                    # Free the raw CPU snapshot to avoid holding large pinned buffers indefinitely.
                    meta._k_host = None
                    meta._v_host = None
                    meta._kv_host = None
                    meta.kv_bytes = 0
                    meta.seq_len = 0
                    meta.tier = "GPU"
                    restored = False

                meta.state = SessionKVState.RUNNABLE
                meta.last_access = time.time()

                if restored:
                    logger.info(f"Session {session_id} restored and marked as RUNNABLE")
                else:
                    logger.info(f"Session {session_id} marked as RUNNABLE")

                return ToolEndResult(
                    ok=True,
                    session_id=session_id,
                    state=meta.state.value,
                    epoch=meta.epoch,
                )
        
        except Exception as e:
            logger.exception(f"Unexpected error in tool_end for session {session_id}")
            return ToolEndResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error=f"Internal error: {str(e)}",
            )
    
    def cleanup_session(self, session_id: str):
        """Clean up all resources for a session.
        
        This method is designed to be robust - it will attempt to clean up
        all resources even if individual cleanup steps fail.
        """
        with self.lock:
            if session_id not in self.session_kv_meta:
                return
                
            meta = self.session_kv_meta[session_id]
            cleanup_errors = []
            
            # Clear CPU tensors - each in its own try/except to ensure all are attempted
            try:
                if meta._k_host is not None:
                    del meta._k_host
                    meta._k_host = None
            except Exception as e:
                cleanup_errors.append(f"_k_host: {e}")
                
            try:
                if meta._v_host is not None:
                    del meta._v_host
                    meta._v_host = None
            except Exception as e:
                cleanup_errors.append(f"_v_host: {e}")
                
            try:
                if meta._kv_host is not None:
                    del meta._kv_host
                    meta._kv_host = None
            except Exception as e:
                cleanup_errors.append(f"_kv_host: {e}")

            try:
                if meta._kv_cache_cpu is not None:
                    meta._kv_cache_cpu = None
            except Exception as e:
                cleanup_errors.append(f"_kv_cache_cpu: {e}")
            
            # Always remove from tracking dict
            try:
                del self.session_kv_meta[session_id]
            except Exception as e:
                cleanup_errors.append(f"session_kv_meta removal: {e}")
            
            if cleanup_errors:
                logger.warning(
                    f"Partial cleanup errors for session {session_id}: {cleanup_errors}"
                )
            else:
                logger.info(f"Cleaned up ToolKVManager resources for session {session_id}")


class ToolKVManagerV2:
    """
    Native HiRadixCache integration for tool-calling KV management.
    
    This version uses HiRadixCache primitives instead of shadow copy:
    - offload_for_tool() for GPU→CPU backup + eviction
    - preload_for_tool() for CPU→GPU restoration
    - protect_host()/release_host() for semantic locking
    
    Benefits:
    - Preserves Radix Attention prefix sharing
    - Leverages SGLang's existing eviction policies
    - No duplicate data management
    
    Requirements:
    - HiRadixCache must be enabled (--enable-hierarchical-cache)
    """
    
    # Session TTL: 1 hour (prevents memory leaks if agent crashes)
    SESSION_TTL_SECONDS = 3600
    
    def __init__(self, scheduler: "Scheduler"):
        self.scheduler = scheduler
        self.lock = threading.RLock()
        
        # Session to last radix node mapping
        self.session_to_last_node: Dict[str, TreeNode] = {}
        
        # Track offload timestamps for TTL-based cleanup
        self.offload_timestamps: Dict[str, float] = {}
        
        # Session metadata (for API compatibility)
        self.session_meta: Dict[str, SessionKVMeta] = {}
        
        # Check if HiRadixCache is available
        self._hicache_available = self._check_hicache()
        
        logger.info(
            f"ToolKVManagerV2 initialized. HiCache enabled: {self._hicache_available}"
        )
    
    def _check_hicache(self) -> bool:
        """Check if HiRadixCache is available."""
        tree_cache = getattr(self.scheduler, "tree_cache", None)
        if tree_cache is None:
            return False
        # Check for HiRadixCache-specific methods
        return hasattr(tree_cache, "offload_for_tool") and hasattr(tree_cache, "preload_for_tool")
    
    def _get_hicache(self):
        """Get HiRadixCache instance or None."""
        if not self._hicache_available:
            return None
        return getattr(self.scheduler, "tree_cache", None)
    
    def _get_or_create_meta(self, session_id: str) -> SessionKVMeta:
        """Get or create session metadata."""
        if session_id not in self.session_meta:
            self.session_meta[session_id] = SessionKVMeta(session_id=session_id)
        return self.session_meta[session_id]
    
    def _find_session_node(
        self, session_id: str, rid: Optional[str] = None
    ) -> Optional[TreeNode]:
        """
        Find the radix tree node for a session.
        
        First checks the cached mapping, then falls back to searching
        through the session's request history.
        """
        # Check cached mapping first
        if session_id in self.session_to_last_node:
            node = self.session_to_last_node[session_id]
            # Validate node is still valid (not garbage collected)
            if node is not None and hasattr(node, 'id'):
                return node
        
        # Fall back to session-based lookup
        session = getattr(self.scheduler, "sessions", {}).get(session_id)
        if session is None:
            return None
        
        req_nodes = getattr(session, "req_nodes", None)
        if not req_nodes:
            return None
        
        # Find the request's last_node
        if rid is not None:
            node_info = req_nodes.get(rid)
            if node_info is not None:
                req = getattr(node_info, "req", None)
                if req is not None:
                    return getattr(req, "last_node", None)
        
        # Find the most recent request's last_node
        for node_info in reversed(list(req_nodes.values())):
            req = getattr(node_info, "req", None)
            if req is not None:
                last_node = getattr(req, "last_node", None)
                if last_node is not None:
                    return last_node
        
        return None
    
    def _estimate_kv_bytes(self, num_tokens: int) -> int:
        """Estimate KV cache size in bytes."""
        try:
            kv_cache = self.scheduler.token_to_kv_pool_allocator.get_kvcache()
            if hasattr(kv_cache, 'k_buffer') and len(kv_cache.k_buffer) > 0:
                # MHA style: k_buffer + v_buffer
                k_shape = kv_cache.k_buffer[0].shape
                num_layers = len(kv_cache.k_buffer)
                bytes_per_layer = k_shape[1:].numel() * kv_cache.k_buffer[0].element_size()
                return num_tokens * num_layers * bytes_per_layer * 2  # k + v
            elif hasattr(kv_cache, 'kv_buffer') and len(kv_cache.kv_buffer) > 0:
                # MLA style: kv_buffer
                kv_shape = kv_cache.kv_buffer[0].shape
                num_layers = len(kv_cache.kv_buffer)
                bytes_per_layer = kv_shape[1:].numel() * kv_cache.kv_buffer[0].element_size()
                return num_tokens * num_layers * bytes_per_layer
        except Exception:
            pass
        # Fallback: rough estimate (FP16, 32 layers, 128 heads * 128 dim)
        return num_tokens * 32 * 128 * 128 * 2 * 2
    
    def tool_start(
        self,
        session_id: str,
        mode: str = "cpu",
        rid: Optional[str] = None,
    ) -> ToolStartResult:
        """
        Handle TOOL_START request: offload KV to CPU using HiRadixCache.
        
        Flow:
        1. Find the session's last node in the radix tree
        2. Call offload_for_tool() to backup and evict GPU memory
        3. The node remains in the tree but with value=None (evicted)
        4. CPU memory is protected via protect_host()
        
        Args:
            session_id: Session to pause
            mode: Offload mode (only "cpu" supported)
            rid: Optional specific request ID
        """
        try:
            if mode != "cpu":
                return ToolStartResult(
                    ok=False,
                    session_id=session_id,
                    state="ERROR",
                    error=f"Unsupported mode: {mode}. Only 'cpu' is supported."
                )
            
            with self.lock:
                tree_cache = self._get_hicache()
                if tree_cache is None:
                    return ToolStartResult(
                        ok=False,
                        session_id=session_id,
                        state="ERROR",
                        error="HiRadixCache not enabled. Use --enable-hierarchical-cache"
                    )
                
                # Find the session's last node
                node = self._find_session_node(session_id, rid)
                if node is None:
                    return ToolStartResult(
                        ok=False,
                        session_id=session_id,
                        state="ERROR",
                        error=f"Session {session_id} not found or has no radix node"
                    )
                
                # Check if already offloaded
                if session_id in self.offload_timestamps:
                    meta = self.session_meta.get(session_id)
                    if meta and meta.state == SessionKVState.OFFLOADED_TO_CPU:
                        return ToolStartResult(
                            ok=True,
                            session_id=session_id,
                            state=meta.state.value,
                            kv_bytes=meta.kv_bytes,
                            epoch=meta.epoch,
                        )
                
                # Offload using HiRadixCache primitive
                num_tokens = tree_cache.offload_for_tool(node, protect=True)
                if num_tokens == 0:
                    return ToolStartResult(
                        ok=False,
                        session_id=session_id,
                        state="ERROR",
                        error="Failed to offload: node may be locked or already evicted without backup"
                    )
                
                # Track session
                self.session_to_last_node[session_id] = node
                self.offload_timestamps[session_id] = time.time()
                
                # Update metadata
                meta = self._get_or_create_meta(session_id)
                meta.state = SessionKVState.OFFLOADED_TO_CPU
                meta.tier = "CPU"
                meta.epoch += 1
                meta.last_access = time.time()
                meta.seq_len = num_tokens
                meta.kv_bytes = self._estimate_kv_bytes(num_tokens)
                meta.rid = rid
                
                logger.info(
                    f"Offloaded session {session_id} to CPU via HiRadixCache: "
                    f"tokens={num_tokens}, node_id={node.id}, epoch={meta.epoch}"
                )
                
                return ToolStartResult(
                    ok=True,
                    session_id=session_id,
                    state=meta.state.value,
                    kv_bytes=meta.kv_bytes,
                    epoch=meta.epoch,
                )
        
        except Exception as e:
            logger.exception(f"Unexpected error in tool_start for session {session_id}")
            return ToolStartResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error=f"Internal error: {str(e)}",
            )
    
    def tool_end(
        self,
        session_id: str,
        epoch: int,
        tool_result: Optional[str] = None,
    ) -> ToolEndResult:
        """
        Handle TOOL_END request: restore KV from CPU to GPU using HiRadixCache.
        
        Flow:
        1. Validate epoch
        2. Call preload_for_tool() to restore CPU→GPU
        3. Release CPU protection via release_host()
        
        Args:
            session_id: Session to resume
            epoch: Expected epoch (must match)
            tool_result: Optional tool output (stored for future use)
        """
        try:
            with self.lock:
                if session_id not in self.session_to_last_node:
                    return ToolEndResult(
                        ok=False,
                        session_id=session_id,
                        state="ERROR",
                        error=f"Session {session_id} not found in offload tracking"
                    )
                
                meta = self.session_meta.get(session_id)
                if meta and meta.epoch != epoch:
                    return ToolEndResult(
                        ok=False,
                        session_id=session_id,
                        state="ERROR",
                        error=f"Epoch mismatch: expected {epoch}, got {meta.epoch}"
                    )
                
                tree_cache = self._get_hicache()
                if tree_cache is None:
                    return ToolEndResult(
                        ok=False,
                        session_id=session_id,
                        state="ERROR",
                        error="HiRadixCache not enabled"
                    )
                
                node = self.session_to_last_node[session_id]
                
                # Store tool result for future use
                if tool_result and meta:
                    meta._pending_tool_result = tool_result
                
                # Preload using HiRadixCache primitive
                success = tree_cache.preload_for_tool(node, release=True)
                if not success:
                    return ToolEndResult(
                        ok=False,
                        session_id=session_id,
                        state="ERROR",
                        epoch=epoch,
                        error="Failed to preload: OOM or node has no backup"
                    )
                
                # Cleanup tracking
                del self.session_to_last_node[session_id]
                del self.offload_timestamps[session_id]
                
                # Update metadata
                if meta:
                    meta.state = SessionKVState.RUNNABLE
                    meta.tier = "GPU"
                    meta.last_access = time.time()
                
                logger.info(
                    f"Preloaded session {session_id} to GPU via HiRadixCache: "
                    f"node_id={node.id}, epoch={epoch}"
                )
                
                return ToolEndResult(
                    ok=True,
                    session_id=session_id,
                    state="RUNNABLE",
                    epoch=epoch,
                )
        
        except Exception as e:
            logger.exception(f"Unexpected error in tool_end for session {session_id}")
            return ToolEndResult(
                ok=False,
                session_id=session_id,
                state="ERROR",
                error=f"Internal error: {str(e)}",
            )
    
    def cleanup_stale_sessions(self) -> int:
        """
        Clean up sessions that have exceeded TTL.
        
        This prevents memory leaks if the agent crashes and never sends tool_end.
        Should be called periodically (e.g., every minute).
        
        Returns:
            Number of sessions cleaned up
        """
        now = time.time()
        stale_sessions = []
        
        with self.lock:
            for session_id, ts in self.offload_timestamps.items():
                if now - ts > self.SESSION_TTL_SECONDS:
                    stale_sessions.append(session_id)
            
            for session_id in stale_sessions:
                logger.warning(
                    f"Session {session_id} exceeded TTL ({self.SESSION_TTL_SECONDS}s), "
                    f"cleaning up to prevent memory leak"
                )
                try:
                    node = self.session_to_last_node.get(session_id)
                    if node is not None and node.host_ref_counter > 0:
                        node.release_host()
                        logger.info(f"Released host lock for stale session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to release host for session {session_id}: {e}")
                
                self.session_to_last_node.pop(session_id, None)
                self.offload_timestamps.pop(session_id, None)
                meta = self.session_meta.pop(session_id, None)
                if meta:
                    meta.state = SessionKVState.INVALID
        
        return len(stale_sessions)
    
    def cleanup_session(self, session_id: str):
        """
        Clean up a specific session.
        
        Called when a session is closed to release any held resources.
        """
        with self.lock:
            if session_id in self.session_to_last_node:
                try:
                    node = self.session_to_last_node[session_id]
                    if node is not None and node.host_ref_counter > 0:
                        node.release_host()
                except Exception as e:
                    logger.warning(f"Failed to release host for session {session_id}: {e}")
                
                del self.session_to_last_node[session_id]
            
            self.offload_timestamps.pop(session_id, None)
            self.session_meta.pop(session_id, None)
            logger.info(f"Cleaned up ToolKVManagerV2 resources for session {session_id}")
    
    def get_kv_meta(self, session_id: str) -> Optional[SessionKVMeta]:
        """Get KV metadata for a session."""
        with self.lock:
            return self.session_meta.get(session_id)
    
    def list_active_requests(self) -> list:
        """List all active requests (for debugging)."""
        # Delegate to scheduler's running state
        requests = []
        
        if self.scheduler.running_batch:
            for req in self.scheduler.running_batch.reqs:
                requests.append({
                    "rid": req.rid,
                    "session_id": getattr(req, 'session_id', None),
                    "location": "running_batch",
                    "seq_len": len(req.origin_input_ids) + len(req.output_ids),
                    "output_len": len(req.output_ids),
                })
        
        return requests
    
    def update_session_node(self, session_id: str, node: TreeNode):
        """
        Update the session-to-node mapping.
        
        Called after a request finishes to track the latest radix node.
        """
        with self.lock:
            self.session_to_last_node[session_id] = node
    
    @property
    def offloaded_session_count(self) -> int:
        """Number of sessions currently offloaded."""
        return len(self.offload_timestamps)
    
    @property
    def session_kv_meta(self) -> Dict[str, SessionKVMeta]:
        """Compatibility property for old API."""
        return self.session_meta
