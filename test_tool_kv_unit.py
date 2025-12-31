import types

from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.managers.tool_kv_manager import ToolKVManagerV2


class DummyNode:
    def __init__(self):
        self.parent = None
        self.children = {}
        self.host_ref_counter = 0
        self.lock_ref = 0


def test_collect_leaves_skips_host_locked():
    rc = RadixCache.__new__(RadixCache)
    rc.root_node = TreeNode.__new__(TreeNode)
    rc.root_node.children = {}

    leaf_allowed = TreeNode.__new__(TreeNode)
    leaf_allowed.children = {}
    leaf_allowed.lock_ref = 0
    leaf_allowed.host_ref_counter = 0

    leaf_blocked = TreeNode.__new__(TreeNode)
    leaf_blocked.children = {}
    leaf_blocked.lock_ref = 0
    leaf_blocked.host_ref_counter = 1

    rc.root_node.children["a"] = leaf_allowed
    leaf_allowed.parent = rc.root_node
    rc.root_node.children["b"] = leaf_blocked
    leaf_blocked.parent = rc.root_node

    leaves = rc._collect_leaves()
    assert leaf_allowed in leaves
    assert leaf_blocked not in leaves


def test_preload_for_tool_releases_host_on_failure():
    # Create HiRadixCache without running its __init__
    hc = HiRadixCache.__new__(HiRadixCache)
    hc._load_back_lock = types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda *args: False)
    hc.load_back = lambda node: None  # force failure
    hc.loading_check = lambda: None

    node = TreeNode.__new__(TreeNode)
    node.evicted = True
    node.backuped = True
    node.host_ref_counter = 1
    released = {}

    def release_host():
        released["called"] = True
    node.release_host = release_host

    ok = HiRadixCache.preload_for_tool(hc, node, release=True)
    assert ok is False
    assert released.get("called") is True


def test_tool_kv_manager_v2_drops_stale_cached_node():
    mgr = ToolKVManagerV2.__new__(ToolKVManagerV2)
    mgr.session_to_last_node = {}
    mgr.session_meta = {}
    mgr.offload_timestamps = {}
    mgr.lock = None
    mgr._hicache_available = True
    mgr.scheduler = types.SimpleNamespace(sessions={})

    stale = TreeNode.__new__(TreeNode)
    stale.parent = None  # detached
    stale.value = None
    stale.host_value = None
    mgr.session_to_last_node["s"] = stale

    node = mgr._find_session_node("s")
    assert node is None
    assert "s" not in mgr.session_to_last_node
