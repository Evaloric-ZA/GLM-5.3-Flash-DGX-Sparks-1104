@ -0,0 +1,401 @@
//! TensorFold glm5_next/cuda/dflash2_multi.py:_block_compute/_taps_compute/_layer.
//! Ash Hart and TensorFold contributors; jschmied CUDA graph/launch patterns.
//! Target embedding is read-only; all writable buffers and KV belong to the drafter.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("cuda_matmuls.zig");
const weights = @import("cuda_dflash_weights.zig");
const config = @import("cuda_dflash_config.zig");
const Tri = @import("cuda_triton.zig").Tri;
const Torch = @import("cuda_torch_ops.zig").Torch;
const Coll = @import("flashnext").mirror.Coll;
const base = @import("cuda_torch_ops.zig").shared;
const stage_mod = @import("cuda_stage.zig");
const Stager = stage_mod.Stager;
/// Pinned readback layout: the tap graph's invalid flag, the block graph's, the candidates, the projection.
const pin_tap = 0;
const pin_block = 16;
const pin_cand = 64;
const pin_proj = pin_cand + 4 * 8 * 7 * 32 * 4;
const pin_bytes = pin_proj + 8 * 7 * 256 * 4;
pub const Forward = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    cfg: config.Config,
    rank: config.Rank,
    streams: usize,
    w: weights.Weights,
    embed: u64,
    tri: Tri,
    th: Torch,
    entries: *const mm.Entries,
    comm: ?Coll,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    selector: cuda.cublaslt.Linear,
    selector_ws: u64,
    selector_ws_len: usize,
    inv: u64,
    ids: u64,
    rowpos: u64,
    rowslots: u64,
    indices: u64,
    bpos: u64,
    bslots: u64,
    x: u64,
    normed: u64,
    dyn: u64,
    conv: u64,
    prod: u64,
    q: u64,
    k: u64,
    v: u64,
    attn: u64,
    gu: u64,
    act: u64,
    sums: u64,
    part: u64,
    rowpart: u64,
    gathered: u64,
    rowout: u64,
    tap_in: u64,
    cos: u64,
    sin: u64,
    h: u64,
    hn: u64,
    logits: u64,
    logitsf: u64,
    vals: u64,
    local: u64,
    top_scratch: u64,
    pack_buf: u64,
    candidates: u64,
    projbf: u64,
    projection: u64,
    invalid: u64,
    kc: [5]u64,
    vc: [5]u64,
    blocks: [8]?cuda.graph.Exec = @splat(null),
    taps: [8]?cuda.graph.Exec = @splat(null),
    /// Stream-ordered pinned staging and readback (unset with TF_GLM_SYNC_STAGE=1: the blocking calls).
    stager: Stager = .{},
    pin: ?cuda.HostBuffer = null,
    tap_pending: bool = false,
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, cfg: config.Config, rank: config.Rank, streams: usize, w: weights.Weights, embed: u64, tri: Tri, th: Torch, entries: *const mm.Entries, comm: ?Coll, lt: *const cuda.cublaslt.Library) !Forward {
        if (streams < 1 or streams > 8 or (rank.world > 1 and comm == null)) return error.BadDraftStreams;
        if (comm) |c| if (c.world != rank.world or c.rank != rank.rank) return error.BadDraftRank;
        var f: Forward = undefined;
        f.gpa = gpa;
        f.d = d;
        f.stream = stream;
        f.cfg = cfg;
        f.rank = rank;
        f.streams = streams;
        f.w = w;
        f.embed = embed;
        f.tri = tri;
        f.th = th;
        f.entries = entries;
        f.comm = comm;
        f.owned = .empty;
        f.blocks = @splat(null);
