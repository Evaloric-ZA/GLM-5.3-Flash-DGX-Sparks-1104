OpenAI Codex v0.166.0
--------
workdir: /home/zurih/codex-glm53-prefill/r1
model: gpt-6.1-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 01a19db1-8bd3-77d1-a4a9-58215e3fce8d
--------
user
# Question: is a 4,096-row prompt chunk really gated on reworking the DFlash2 drafter?

You are reviewing TensorFold's GLM-5.3-Flash CUDA engine (Zig). `src/` holds the relevant files, copied from a
checkout of upstream `f350cd3e` with this recipe's engine patches 0001-0066 applied (patches 0067-0093 exist but are
not in these copies; none of them changes the constants below). Read-only: do not edit anything. Answer in `answer.md`.

## What I measured on two DGX Sparks (GB10, TP=2, one rank each)

`src/cuda_state.zig` has `pub const prefill_rows = 2048`. `src/cuda_exl3.zig` sets
`chunk_rows = cuda_state.prefill_rows` with a note that a call to the common EXL3 experts decodes **every** expert's
weights, so a prompt pass makes one call a chunk and that decode amortises over the chunk's rows; it records that at
1,024 rows "every trellis tile was decoded twice a pass" and names 4,096 as where upstream PR #1576 puts it.

I changed only that constant to 4,096 (`0094-...patch`) and measured, three cold trials a point, fresh random prompt
content each trial so no prefix cache is hit, `DRAFTER=none` on both sides so the drafter is out of the comparison:

| chunk rows | prefill @23.5K tok | @95K tok | NLL (4 passages) | sha of 10 greedy replies |
| --- | --- | --- | --- | --- |
| 2,048 | 1,626 tok/s | 1,648 tok/s | 0.7523 | 11ceddde19d4 |
| 4,096 | 1,743 tok/s (+7.2%) | 1,757 tok/s (+6.6%) | 0.7523 | 11ceddde19d4 |

So +7%, and bit-exact (identical NLL and identical greedy replies).

With the real config (`DRAFTER=dflash2`, the shipped default) the pair loads, passes a smoke test, and then the first
genuine prefill ends both ranks:

```
error: rank 0: rank 1 stopped (DrafterContextTooWide); this rank stops too
```

`src/dflash.zig` has `MAX_CTX = 2048` ("context rows one ingest takes (a prompt chunk's)") and `WIN = 2048`
("the context window (and a ring's slots)"), with the checks at lines ~282, ~370 and ~385.

## My current conclusion, which I want you to check

I concluded that getting the +7% needs a change to the drafter's ingest path (letting it take a wide chunk in
`MAX_CTX`-sized slices, since it only needs the last `WIN` rows), because `WIN` is the trained drafter's attention
window with rotary tables built for it, and so is not a buffer I can simply widen.

## Specific questions

1. **Is `MAX_CTX` alone the binding constraint?** If `MAX_CTX` were raised to 4,096 and `WIN` left at 2,048, would the
   result be correct, or does `WIN` genuinely bound how many rows of one chunk the drafter can ingest? Trace what
   `MAX_CTX` sizes (`dflash_load.zig` allocations, `gs`) versus what `WIN` governs (which taps/positions are read, the
   ring's slots, rotary). Cite line numbers.
2. **Does a paired prompt step already amortise the expert trellis decode over 4,096 rows?** `src/cuda_prompt_pair.zig`
   (TF_GLM_PROMPT_PAIR, on by default, `default_min_len = 2 * prefill_rows + 1`) makes a prompt step take two chunks.
   Does the EXL3 expert call then cover both chunks' rows in one call, or one call per chunk? If pairing already covers
   two chunks, why would raising `prefill_rows` still give +7%? This is the answer I most want.
3. **Is there a cheaper path to the same win** -- amortising the expert trellis decode over more rows *without*
   changing the chunk size the drafter ingests? For instance, making the experts call span a paired step's two chunks,
   or decoding the trellis once for two consecutive calls.
4. If a drafter-side change is genuinely required, what is the smallest correct one, and what would it risk (draft
   acceptance, `BadPromptContext` at ~385, the ring's slots)?

Be concrete and cite file:line. Say plainly if you think my conclusion is wrong.

codex
I’ll trace the paired prompt and drafter paths through the CUDA code, then assess which changes preserve the existing window and ring behavior. The workspace is read-only, so I’ll provide the findings here if I can’t create `answer.md`.
exec
/bin/bash -lc "pwd && rg -n 'prefill_rows|chunk_rows|MAX_CTX|\\bWIN\\b|DrafterContextTooWide|BadPromptContext|default_min_len|PROMPT_PAIR' src/cuda_state.zig src/cuda_exl3.zig src/dflash.zig src/dflash_load.zig src/cuda_prompt_pair.zig" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
/home/zurih/codex-glm53-prefill/r1
src/dflash_load.zig:200:    try @import("dflash_load_owner.zig").acquire(gpa, dr, &tail, Tail, .{ .book = df.VOCAB * df.RANK, .ctx = df.MAX_CTX, .cand = candidates.host, .hidden = df.MAX_M * df.RANK });
src/dflash_load.zig:228:    const C: u64 = df.MAX_CTX;
src/dflash.zig:12:pub const WIN: i64 = 2048; // the context window (and a ring's slots)
src/dflash.zig:30:pub const MAX_CTX: usize = 2048; // context rows one ingest takes (a prompt chunk's)
src/dflash.zig:35:pub const ring_bytes: u64 = LAYERS * 2 * @as(u64, @intCast(WIN)) * KVW * 2;
src/dflash.zig:137:    gs: u64 = 0, // a 4-bit linear's input group sums [MAX_CTX][5 * D / 64]
src/dflash.zig:227:        return @max(@max(s.first, s.from), end - WIN);
src/dflash.zig:282:        if (N > MAX_CTX) return error.DrafterContextTooWide;
src/dflash.zig:369:    if (start > len or rows > len - start) return error.BadPromptContext;
src/dflash.zig:370:    if (rows > MAX_CTX or row0 > MAX_CTX - rows) return error.DrafterContextTooWide;
src/dflash.zig:371:    const keep_from = len -| @as(usize, @intCast(WIN));
src/dflash.zig:385:    if (start < len -| @as(usize, @intCast(WIN)) or row0 > MAX_CTX or rows > MAX_CTX - row0) return error.BadPromptContext;
src/cuda_exl3.zig:224:/// (it follows cuda_state.prefill_rows, 4,096 with PR #1576). At 1024 every trellis tile was decoded twice a pass for ~28
src/cuda_exl3.zig:226:pub const chunk_rows = @import("cuda_state.zig").prefill_rows;
src/cuda_exl3.zig:256:    const cap: usize = @min(rows, chunk_rows);
src/cuda_exl3.zig:262:    const n: usize = @min(rows, chunk_rows);
src/cuda_exl3.zig:286:    const cap: usize = @min(rows, chunk_rows);
src/cuda_exl3.zig:290:    while (at < rows) : (at += chunk_rows) {
src/cuda_exl3.zig:291:        const n: usize = @min(chunk_rows, rows - at);
src/cuda_exl3.zig:309:    if (rows == 0 or rows > chunk_rows) return null;
src/cuda_exl3.zig:336:    for (1..3 * chunk_rows) |rows| {
src/cuda_exl3.zig:337:        const cap: usize = @min(rows, chunk_rows);
src/cuda_prompt_pair.zig:1://! A prompt step of two chunks in one pass (TF_GLM_PROMPT_PAIR): each chunk's exchanges under the other's sites.
src/cuda_prompt_pair.zig:12:pub const min_rows = fw.prefill_rows / 4;
src/cuda_prompt_pair.zig:16:/// pair is ahead. TF_GLM_PROMPT_PAIR_MIN overrides it in ids (0: pair wherever a step can). Rank 0's choice, so the
src/cuda_prompt_pair.zig:18:pub const default_min_len = 2 * fw.prefill_rows + 1;
src/cuda_prompt_pair.zig:21:    const v = std.c.getenv("TF_GLM_PROMPT_PAIR_MIN") orelse return default_min_len;
src/cuda_prompt_pair.zig:22:    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default_min_len;
src/cuda_prompt_pair.zig:52:/// Whether an engine makes the pair's twin: TF_GLM_PROMPT_PAIR on, and other ranks whose exchanges a pair hides.
src/cuda_prompt_pair.zig:59:    const next = start + fw.prefill_rows;
src/cuda_prompt_pair.zig:77:    const ra = fw.prefill_rows;
src/cuda_prompt_pair.zig:87:    const rb = @min(len - start - fw.prefill_rows, fw.prefill_rows);
src/cuda_prompt_pair.zig:88:    if (!e.pair.?.own_room and !piecesSplit(fw.prefill_rows, rb, e.pieces.count, e.share.world)) return false;
src/cuda_prompt_pair.zig:98:    const ra: usize = fw.prefill_rows;
src/cuda_prompt_pair.zig:100:    const rb: usize = @min(len - start - ra, fw.prefill_rows);
src/cuda_prompt_pair.zig:120:    const keep_from = len -| @as(usize, @intCast(dflash.WIN));
src/cuda_prompt_pair.zig:167:    const C = fw.prefill_rows;
src/cuda_prompt_pair.zig:179:    const C = fw.prefill_rows;
src/cuda_prompt_pair.zig:188:    const C = fw.prefill_rows;
src/cuda_state.zig:19:pub const prefill_rows = 2048;
src/cuda_state.zig:27:pub const index_ring = 2 * prefill_rows;

exec
/bin/bash -lc "nl -ba src/cuda_exl3.zig | sed -n '190,335p'" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
   190	        }
   191	        @field(out, sv) = try L.put("layers.{d}.moe.exl3." ++ sv, .{i});
   192	    }
   193	    out.uniform_gu = uniform(k2s[0 .. 2 * count]);
   194	    out.uniform_d = uniform(k2s[2 * count ..]);
   195	    const tabs = [3][]const i32{ k2s[0..count], k2s[count .. 2 * count], k2s[2 * count ..] };
   196	    if (std.mem.eql(i32, tabs[0], tabs[1]) and std.mem.eql(i32, tabs[0], tabs[2])) { // one table, as a 4-bit layer has it
   197	        try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(tabs[0]));
   198	        out.k2 = try L.put("layers.{d}.moe.exl3.k2", .{i});
   199	        out.k2_up = out.k2;
   200	        out.k2_down = out.k2;
   201	        return out;
   202	    }
   203	    var ptrs: [3]u64 = undefined;
   204	    inline for (tabs, .{ "k2_gate", "k2_up", "k2_down" }, 0..) |tab, name, p| { // a table a distinct projection
   205	        var same: ?u64 = null;
   206	        for (tabs[0..p], ptrs[0..p]) |before, ptr| if (same == null and std.mem.eql(i32, before, tab)) {
   207	            same = ptr;
   208	        };
   209	        ptrs[p] = same orelse blk: {
   210	            try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(tab));
   211	            break :blk try L.put("layers.{d}.moe.exl3." ++ name, .{i});
   212	        };
   213	    }
   214	    out.k2, out.k2_up, out.k2_down = ptrs;
   215	    return out;
   216	}
   217	
   218	/// The selected common EXL3 owner. This file adapts the family's checkpoint shard and slot layout only.
   219	pub const Kernels = cuda.exl3_experts.Experts;
   220	const common = cuda.exl3_experts;
   221	pub const Scratch = struct { picks: u64, xg: u64, xu: u64, z: u64 };
   222	pub const top_k = 8;
   223	/// Rows a call to the common experts takes: each call decodes every expert's weights, so a prompt pass takes one call
   224	/// (it follows cuda_state.prefill_rows, 4,096 with PR #1576). At 1024 every trellis tile was decoded twice a pass for ~28
   225	/// pairs an expert; a pair's sums do not depend on which item holds it, so the bits are the same.
   226	pub const chunk_rows = @import("cuda_state.zig").prefill_rows;
   227	/// The common module's layer: a width table a projection, and the widths the loader found uniform.
   228	pub fn layer(w: Experts, count: usize, dims: usize, width: usize, limit: f32) common.Layer {
   229	    return .{
   230	        .gate = w.gt,
   231	        .up = w.ut,
   232	        .down = w.dt,
   233	        .gate_k2 = w.tables()[0],
   234	        .up_k2 = w.tables()[1],
   235	        .down_k2 = w.tables()[2],
   236	        .suh_g = w.suh_g,
   237	        .suh_u = w.suh_u,
   238	        .svh_g = w.svh_g,
   239	        .svh_u = w.svh_u,
   240	        .suh_d = w.suh_d,
   241	        .svh_d = w.svh_d,
   242	        .experts = count,
   243	        .dims = dims,
   244	        .width = width,
   245	        .codebook = .mcg,
   246	        .limit = limit,
   247	        .act_mode = 0,
   248	        .uniform_k2 = if (w.uniform_gu == w.uniform_d) w.uniform_gu else 0,
   249	        .uniform_k2_gu = w.uniform_gu,
   250	        .uniform_k2_d = w.uniform_d,
   251	        .shared_input = w.shx,
   252	    };
   253	}
   254	/// A chunk's compacted picks ahead of the common scratch, 256-aligned; the usize keeps @min from narrowing to u7.
   255	pub fn picksBytes(rows: usize) usize {
   256	    const cap: usize = @min(rows, chunk_rows);
   257	    return std.mem.alignForward(usize, cap * top_k * 4, 256);
   258	}
   259	/// The scratch `run` takes on `w`'s experts: the prompt kernels' when `prefill` and they take this layer, else the
   260	/// grouped path's (a layer the prompt kernels refuse runs its prompt chunks there, in the same arena).
   261	pub fn scratchFor(rows: usize, dims: usize, width: usize, prefill: bool, w: Experts) [4]usize {
   262	    const n: usize = @min(rows, chunk_rows);
   263	    const l = layer(w, cuda.grouped.max_experts, dims, width, 0);
   264	    // the prompt kernels write the slot rows (Scratch.yrow): no y [pairs][dims] in the arena
   265	    if (prefill and common.prompt.eligible(l)) return .{ picksBytes(rows) + (common.prompt.bytesFor(l, n, top_k, false) catch unreachable), 4, 4, 4 };
   266	    return .{ picksBytes(rows) + (common.scratchBytes(l, n, top_k) catch unreachable), 4, 4, 4 };
   267	}
   268	/// `scratchFor` a 4-bit layer (every projection 8 half-bits), `shx` its shared rotation.
   269	pub fn scratchBytes(rows: usize, dims: usize, width: usize, prefill: bool, shx: bool) [4]usize {
   270	    return scratchFor(rows, dims, width, prefill, .{ .gt = 0, .ut = 0, .dt = 0, .k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .uniform_gu = 8, .uniform_d = 8, .shx = shx });
   271	}
   272	/// `most` raised to `w`'s scratch, entry by entry: buffers every layer's experts run in take the largest.
   273	pub fn mostScratch(most: *[4]usize, rows: usize, dims: usize, width: usize, prefill: bool, w: Experts) void {
   274	    for (most, scratchFor(rows, dims, width, prefill, w)) |*m, v| m.* = @max(m.*, v);
   275	}
   276	/// `height` rows of `width` bytes, `spitch` apart at `src`, to `dpitch` apart at `dst`, in one call.
   277	fn copyRows(stream: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize, what: []const u8) !void {
   278	    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
   279	    try stream.d.check(stream.d.api.cuMemcpy2DAsync_v2(&m, stream.handle), what);
   280	}
   281	pub fn run(k: *const Kernels, ek: *const experts.Kernels, stream: cuda.Stream, w: Experts, sc: Scratch, plan_buf: cuda.grouped.Plan, x: u64, x_stride: usize, picks: u64, slots: usize, rows: usize, count: usize, dims: usize, width: usize, limit: f32, prompt: bool, ey: u64, xd: u64) !void {
   282	    _ = xd;
   283	    if (slots != top_k + 1) return error.UnsupportedTopK;
   284	    if (dims % 256 != 0 or width % 128 != 0) return error.UnreadableExl3;
   285	    const l = layer(w, count, dims, width, limit);
   286	    const cap: usize = @min(rows, chunk_rows);
   287	    const arena = sc.picks + picksBytes(rows);
   288	    const fused = prompt and common.prompt.eligible(l);
   289	    var at: usize = 0;
   290	    while (at < rows) : (at += chunk_rows) {
   291	        const n: usize = @min(chunk_rows, rows - at);
   292	        try copyRows(stream, sc.picks, top_k * 4, picks + at * slots * 4, slots * 4, top_k * 4, n, "EXL3 compact route rows");
   293	        if (fused) { // the down kernel writes this chunk's routed slots of ey straight (slot top_k: the shared expert's)
   294	            var scratch = try common.prompt.scratchFor(arena, l, cap, top_k, false);
   295	            scratch.y = ey + at * slots * dims * 4;
   296	            scratch.yrow = slots;
   297	            try k.prompt.run(k, ek.router, stream, l, x + at * x_stride * 2, x_stride, sc.picks, n, scratch, plan_buf);
   298	            continue;
   299	        }
   300	        const scratch = try common.scratch(arena, l, cap, top_k);
   301	        try k.runSlots(stream, l, x + at * x_stride * 2, x_stride, .bf16, sc.picks, n, scratch);
   302	        try copyRows(stream, ey + at * slots * dims * 4, slots * dims * 4, scratch.y, top_k * dims * 4, top_k * dims * 4, n, "EXL3 routed output slots");
   303	    }
   304	}
   305	
   306	/// A decode window's routed experts after a fused select (glue.selectGroup): the call's common scratch, whose ids,
   307	/// count and members the select fills and whose compact picks are sc.picks. Null when the window takes the old path.
   308	pub fn decodeScratch(w: Experts, sc: Scratch, rows: usize, count: usize, dims: usize, width: usize, limit: f32) !?common.Scratch {
   309	    if (rows == 0 or rows > chunk_rows) return null;
   310	    const l = layer(w, count, dims, width, limit);
   311	    return try common.scratch(sc.picks + picksBytes(rows), l, rows, top_k);
   312	}
   313	
   314	/// The grouped decode path's experts (picks compact in sc.picks, grouped by the fused select): the down partials only;
   315	/// `combineDecode` then writes the layer's output. No copies: the slot rows of `ey` are not used.
   316	pub fn runGrouped(k: *const Kernels, stream: cuda.Stream, w: Experts, sc: Scratch, x: u64, x_stride: usize, rows: usize, count: usize, dims: usize, width: usize, limit: f32) !void {
   317	    if (dims % 256 != 0 or width % 128 != 0) return error.UnreadableExl3;
   318	    const l = layer(w, count, dims, width, limit);
   319	    const scratch = (try decodeScratch(w, sc, rows, count, dims, width, limit)) orelse return error.BadWindow;
   320	    try k.core(stream, l, x, x_stride, .bf16, sc.picks, rows, scratch);
   321	}
   322	
   323	/// The grouped decode path's output: every row's routed slots (from the down partials) and its shared expert's row
   324	/// `shared` (fp32 [rows][dims]) summed with `wts` (`top_k` + 1 a row) in slot order, as glue.combine sums the slot rows.
   325	pub fn combineDecode(k: *const Kernels, stream: cuda.Stream, w: Experts, sc: Scratch, rows: usize, count: usize, dims: usize, width: usize, limit: f32, wts: u64, shared: u64, out: u64) !void {
   326	    const l = layer(w, count, dims, width, limit);
   327	    const scratch = (try decodeScratch(w, sc, rows, count, dims, width, limit)) orelse return error.BadWindow;
   328	    try k.combine(stream, l, sc.picks, rows, scratch, wts, top_k + 1, shared, out);
   329	}
   330	
   331	test {
   332	    _ = @import("cuda_exl3_test.zig");
   333	}
   334	
   335	test "a chunk's compacted picks end before the common scratch at every row count" {

exec
/bin/bash -lc "nl -ba src/dflash_load.zig | sed -n '185,290p'" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
   185	            lw.down = try l.linearB(&.{down}, &.{D}, FF, .{ 0, 0, rank * ff, ff });
   186	        } else {
   187	            lw.gu = try l.linear(&.{ gate, up }, &.{ FF, FF }, D, null);
   188	            lw.down = try l.linearB(&.{down}, &.{D}, FF, null);
   189	        }
   190	        lw.a_proj = try l.linear(&.{try layerName(&nb[1], i, "attention_conv.kernel_projection.weight")}, &.{KVW}, D, null);
   191	        lw.m_proj = try l.linear(&.{try layerName(&nb[1], i, "mlp_conv.kernel_projection.weight")}, &.{KVW}, D, null);
   192	    }
   193	    dr.fc = try l.linear(&.{"fc.weight"}, &.{D}, 5 * D, null);
   194	    dr.hproj = try l.linearB(&.{"candidate_selector.hidden_projection.weight"}, &.{df.RANK}, D, null);
   195	    dr.hnorm = try l.raw("hidden_norm.weight", D * 2);
   196	    dr.norm = try l.raw("norm.weight", D * 2);
   197	    if (l.at > total) return error.DrafterLayout;
   198	    var tail: Tail = .{ .loader = &l, .head_n = head.n };
   199	    const candidates = @import("dflash_candidates.zig").Layout.init(world, df.MAX_M, df.CW);
   200	    try @import("dflash_load_owner.zig").acquire(gpa, dr, &tail, Tail, .{ .book = df.VOCAB * df.RANK, .ctx = df.MAX_CTX, .cand = candidates.host, .hidden = df.MAX_M * df.RANK });
   201	    if (std.c.getenv("TF_GLM_DFLASH2_LOG")) |v| dr.log = std.fmt.parseInt(u64, std.mem.span(v), 10) catch 0;
   202	    if (std.c.getenv("TF_GLM_DFLASH2_PROF") != null) dr.prof = io;
   203	    std.log.info("dflash2: drafter from {s} (CC BY-NC-ND 4.0: non-commercial use only), linears at {d} bits, {d:.2} GiB on the device", .{ dir, if (bits == 0) 16 else bits, @as(f64, @floatFromInt(l.at)) / (1 << 30) });
   204	    return dr;
   205	}
   206	
   207	const Tail = struct {
   208	    loader: *Loader,
   209	    head_n: usize,
   210	
   211	    pub fn readBook(t: *Tail, i: usize, dst: []u16) !void {
   212	        const name = if (i == 0) "candidate_selector.predecessor_codebook" else "candidate_selector.successor_codebook";
   213	        @memcpy(std.mem.sliceAsBytes(dst), try t.loader.read(name, dst.len * 2));
   214	    }
   215	
   216	    pub fn scratch(t: *Tail, dr: *Drafter) !void {
   217	        try carve(dr, t.loader.w.d, t.head_n);
   218	    }
   219	
   220	    pub fn freeScratch(_: *Tail, dr: *Drafter) void {
   221	        dr.scratch.free();
   222	    }
   223	};
   224	
   225	/// The drafter's scratch, every field one carved piece of one buffer.
   226	fn carve(dr: *Drafter, d: *const cuda.Driver, head_n: usize) !void {
   227	    const M: u64 = df.MAX_M;
   228	    const C: u64 = df.MAX_CTX;
   229	    const chunks: u64 = (head_n + 4095) / 4096;
   230	    var part: u64 = 16;
   231	    for (dr.layers) |l| {
   232	        for ([_]df.Lin{ l.qkv, l.kv, l.o, l.gu, l.a_proj, l.m_proj }) |w| if (w == .bf16) {
   233	            part = @max(part, mm.partBytes(64, w.bf16));
   234	        };
   235	        part = @max(part, mm.partBytes(64, l.down));
   236	    }
   237	    if (dr.fc == .bf16) part = @max(part, mm.partBytes(64, dr.fc.bf16));
   238	    part = @max(part, mm.partBytes(64, dr.hproj));
   239	    const fields = .{
   240	        .{ "x", M * D * 2 },                                                                     .{ "y", M * D * 2 },                        .{ "y2", M * D * 2 },                .{ "dyn", M * KVW * 2 },
   241	        .{ "qkv", M * df.QKV * 2 },                                                              .{ "q", M * D * 2 },                        .{ "kb", M * KVW * 2 },              .{ "vb", M * KVW * 2 },
   242	        .{ "att", M * D * 2 },                                                                   .{ "a", M * D * 2 },                        .{ "gu", M * 2 * FF * 2 },           .{ "act", M * FF * 2 },
   243	        .{ "hn", M * D * 2 },                                                                    .{ "xs", M * (D / 64) * 4 },                .{ "g", M * df.RANK * 4 },           .{ "logits", M * head_n * 2 },
   244	        .{ "part", part },                                                                       .{ "gs", C * 5 * (D / 64) * 4 },            .{ "sums", 2 * M * D * 4 },          .{ "cin", C * df.TAP_ROW },
   245	        .{ "c1", C * D * 2 },                                                                    .{ "ctx", C * D * 2 },                      .{ "ckv", C * 2 * KVW * 2 },         .{ "taps_step", df.STEP_ROWS * df.TAP_ROW },
   246	        .{ "taps_prefill", C * df.TAP_ROW },                                                     .{ "ids", M * 4 },                          .{ "bpos", M * 4 },                  .{ "rings", df.MAX_S * 8 },
   247	        .{ "starts", df.MAX_S * 8 },                                                             .{ "froms", df.MAX_S * 8 },                 .{ "cpos", C * 8 },                  .{ "crings", C * 8 },
   248	        .{ "gtk", @import("dflash_candidates.zig").Layout.init(dr.world, M, df.CW).device * 4 }, .{ "pv", M * chunks * df.TOPK * 4 },        .{ "pi", M * chunks * df.TOPK * 4 }, .{ "pm", M * df.QH * df.CH * 4 },
   249	        .{ "pl", M * df.QH * df.CH * 4 },                                                        .{ "pacc", M * df.QH * df.CH * df.HD * 4 },
   250	    };
   251	    var at: u64 = 0;
   252	    inline for (fields) |f| {
   253	        @field(dr, f[0]) = at;
   254	        at += std.mem.alignForward(u64, f[1], 256);
   255	    }
   256	    dr.scratch = try cuda.DeviceBuffer.alloc(d, at);
   257	    errdefer dr.scratch.free();
   258	    inline for (fields) |f| @field(dr, f[0]) += dr.scratch.ptr;
   259	    try d.check(d.api.cuMemsetD8_v2(dr.scratch.ptr, 0, at), "cuMemsetD8");
   260	}
   261	
   262	/// Device bytes the drafter's scratch takes beside its weights (the memory budget's share, before loading).
   263	pub fn checkConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
   264	    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
   265	    defer gpa.free(path);
   266	    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
   267	    defer gpa.free(text);
   268	    const p = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
   269	    defer p.deinit();
   270	    const o = p.value.object;
   271	    const dc = (o.get("dflash_config") orelse return error.NotDFlash2).object;
   272	    const ok = o.get("hidden_size").?.integer == D and o.get("num_hidden_layers").?.integer == df.LAYERS and
   273	        o.get("num_attention_heads").?.integer == 32 and o.get("num_key_value_heads").?.integer == 8 and
   274	        o.get("intermediate_size").?.integer == FF and o.get("vocab_size").?.integer == df.VOCAB and
   275	        dc.get("block_size").?.integer == df.BLOCK and dc.get("mask_token_id").?.integer == df.MASK and
   276	        dc.get("selector_top_k").?.integer == df.TOPK and dc.get("selector_rank").?.integer == df.RANK and
   277	        dc.get("conv_kernel_size").?.integer == 2 and dc.get("conv_group_size").?.integer == 16;
   278	    if (!ok) return error.DrafterConfigUnexpected;
   279	    const taps = dc.get("target_layer_ids").?.array.items;
   280	    if (taps.len != taps_mod.layers.len) return error.DrafterConfigUnexpected;
   281	    for (taps, taps_mod.layers) |t, w| if (t.integer != w) return error.DrafterConfigUnexpected;
   282	}

exec
/bin/bash -lc 'nl -ba src/dflash.zig' in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
     1	//! DFlash2 drafts (--drafter): incoai/GLM-5.3-Flash-DFlash2 (CC BY-NC-ND 4.0, read unmodified), after z-lab (MIT).
     2	const std = @import("std");
     3	const cuda = @import("cuda");
     4	const mm = @import("cuda_mm.zig");
     5	const q4 = @import("cuda_q4.zig");
     6	const taps_mod = @import("cuda_taps.zig");
     7	const select = @import("dflash_select.zig");
     8	const block = @import("dflash_block.zig");
     9	
    10	pub const BLOCK: usize = 8;
    11	pub const MASK: u32 = 154856;
    12	pub const WIN: i64 = 2048; // the context window (and a ring's slots)
    13	pub const LAYERS: usize = 5;
    14	pub const D: usize = 4096;
    15	pub const KVW: usize = 1024;
    16	pub const QKV: usize = 4096 + 2 * KVW;
    17	pub const FF: usize = 12288;
    18	pub const RANK: usize = 256;
    19	pub const TOPK: usize = 16;
    20	pub const THETA: f32 = 10000;
    21	pub const EPS: f32 = 1e-5;
    22	pub const SCALE: f32 = 0.08838834764831845; // 128^-0.5
    23	pub const VOCAB: usize = 154880;
    24	pub const QH: usize = 32;
    25	pub const HD: usize = 128;
    26	pub const CH: usize = 16; // key chunks of the chunked attention (df_attn_part)
    27	pub const CW: usize = 2 * TOPK; // a row's gathered ints: 16 ids, then 16 values' bits
    28	pub const MAX_S: usize = 16; // streams one block pass drafts for
    29	pub const MAX_M: usize = MAX_S * BLOCK; // block rows at most
    30	pub const MAX_CTX: usize = 2048; // context rows one ingest takes (a prompt chunk's)
    31	pub const STEP_ROWS: usize = 32; // a verify's tap rows (the shared windows' rows at most)
    32	pub const TAP_ROW: usize = taps_mod.layers.len * D * 2; // bytes of one row's taps
    33	
    34	/// Device bytes a stream's context ring takes (K and V of every layer).
    35	pub const ring_bytes: u64 = LAYERS * 2 * @as(u64, @intCast(WIN)) * KVW * 2;
    36	
    37	/// Device bytes of the loader's raw scratch: a 4-bit linear's words, scales and biases, for the largest at `world`.
    38	pub fn rawBytes(bits: u8, world: usize) usize {
    39	    const ff: usize = if (world == 2) FF / 2 else FF; // the MLP width a rank holds, as dflash_load splits it
    40	    const nk: usize = @max(5 * D * D, 2 * ff * D); // fc, or gate and up stacked
    41	    return if (bits == 4) nk / 2 + 2 * nk / 32 + 1024 else 256;
    42	}
    43	
    44	/// glm_dflash's functions.
    45	pub const Kernels = struct {
    46	    tap: cuda.Function,
    47	    block_qkv: cuda.Function,
    48	    ctx_kv: cuda.Function,
    49	    conv: cuda.Function,
    50	    swiglu: cuda.Function,
    51	    attn_part: cuda.Function,
    52	    attn_merge: cuda.Function,
    53	    sum2: cuda.Function,
    54	    top16_part: cuda.Function,
    55	    top16_merge: cuda.Function,
    56	    gsum: cuda.Function,
    57	
    58	    pub fn load(m: cuda.Module) !Kernels {
    59	        var k: Kernels = undefined;
    60	        inline for (@typeInfo(Kernels).@"struct".field_names) |n| @field(k, n) = try m.function("df_" ++ n);
    61	        return k;
    62	    }
    63	};
    64	
    65	/// A drafter linear: 4 bits (--drafter-bits 4, the default; they only propose) or bf16 tiled for glm_mm.
    66	pub const Lin = union(enum) { q4: q4.Q4, bf16: mm.Tiled };
    67	
    68	/// down and the selector's projection stay bf16 (their fp32 outputs are gathered or read on the host).
    69	pub const LayerW = struct { in_norm: u64, post_norm: u64, qn: u64, kn: u64, a_base: u64, m_base: u64, qkv: Lin, kv: Lin, o: Lin, gu: Lin, down: mm.Tiled, a_proj: Lin, m_proj: Lin };
    70	
    71	/// A ring's `first` before any row of its sequence is in it.
    72	pub const no_rows: i64 = std.math.maxInt(i64);
    73	
    74	/// A stream's drafter state by slot: its ring, where its context starts, and its last verify's tap rows.
    75	pub const Slot = struct {
    76	    ring: ?cuda.DeviceBuffer = null,
    77	    from: i64 = 0, // the first position the ring holds for this sequence
    78	    first: i64 = no_rows, // prompt rows held: the first of the run in the ring that ends at `end` (no_rows: none yet)
    79	    end: i64 = 0,
    80	    v_step: u64 = 0, // the verify that wrote its rows (Drafter.verifies then)
    81	    v_row0: usize = 0,
    82	    v_rows: usize = 0,
    83	    v_pos0: i64 = 0,
    84	};
    85	
    86	/// One stream's request: `n` rows of its last verify into its context first (0: none), the anchor, rank 0's `out`.
    87	pub const Req = struct { slot: u32, pos: i64, n: usize, anchor: u32, out: ?[]u32 = null, probs: ?[]f64 = null, keyed: ?select.Keyed = null };
    88	
    89	/// Context runs: `n` tap rows at `taps` (consecutive), positions pos0.., into `ring`.
    90	pub const Run = struct { ring: u64, taps: u64, n: usize, pos0: i64 };
    91	
    92	/// Every rank runs the ingest and the block pass on its half of the MLP and head; rank 0 selects the drafts.
    93	pub const Drafter = struct {
    94	    gpa: std.mem.Allocator,
    95	    d: *const cuda.Driver,
    96	    module: cuda.Module,
    97	    k: Kernels,
    98	    weights: cuda.DeviceBuffer,
    99	    layers: [LAYERS]LayerW = undefined,
   100	    fc: Lin = undefined,
   101	    hproj: mm.Tiled = undefined,
   102	    hnorm: u64 = 0,
   103	    norm: u64 = 0,
   104	    pred: []u16 = &.{}, // the selector's codebooks, bf16 [VOCAB][RANK] (host: it reads 16 rows a draft)
   105	    succ: []u16 = &.{},
   106	    split: bool, // the MLP split across two ranks (gate and up by rows, down by K)
   107	    ff: usize, // the MLP width this rank holds
   108	    rank: usize,
   109	    world: usize,
   110	    temp: f64 = 1.0, // the selector's softmax temperature for the drafts' chances
   111	    scratch: cuda.DeviceBuffer = undefined,
   112	    slots: std.ArrayList(Slot) = .empty,
   113	    verifies: u64 = 0, // verifies so far: a slot's rows are still in taps_step while its v_step is the latest
   114	    lost: u64 = 0, // drafts whose verify rows were gone (their context restarted)
   115	    calls: u64 = 0,
   116	    log: u64 = 0, // TF_GLM_DFLASH2_LOG=n: the first n rounds' drafts logged (rank 0)
   117	    prof: ?std.Io = null, // TF_GLM_DFLASH2_PROF=1: each phase synchronized and timed, logged every 64 rounds
   118	    phase_ms: [4]f64 = @splat(0), // ingest, tables, block pass, selector
   119	    // device scratch (carved by dflash_load)
   120	    x: u64 = 0,
   121	    y: u64 = 0,
   122	    y2: u64 = 0,
   123	    dyn: u64 = 0,
   124	    qkv: u64 = 0,
   125	    q: u64 = 0,
   126	    kb: u64 = 0,
   127	    vb: u64 = 0,
   128	    att: u64 = 0,
   129	    a: u64 = 0,
   130	    gu: u64 = 0,
   131	    act: u64 = 0,
   132	    hn: u64 = 0,
   133	    xs: u64 = 0,
   134	    g: u64 = 0,
   135	    logits: u64 = 0,
   136	    part: u64 = 0, // the linears' K-slice partials
   137	    gs: u64 = 0, // a 4-bit linear's input group sums [MAX_CTX][5 * D / 64]
   138	    sums: u64 = 0, // the split down's gathered fp32 partials [2][MAX_M][D]
   139	    cin: u64 = 0,
   140	    c1: u64 = 0,
   141	    ctx: u64 = 0,
   142	    ckv: u64 = 0,
   143	    taps_step: u64 = 0,
   144	    taps_prefill: u64 = 0,
   145	    ids: u64 = 0,
   146	    bpos: u64 = 0,
   147	    rings: u64 = 0,
   148	    starts: u64 = 0,
   149	    froms: u64 = 0,
   150	    cpos: u64 = 0,
   151	    crings: u64 = 0,
   152	    tk: u64 = 0,
   153	    gtk: u64 = 0,
   154	    pv: u64 = 0,
   155	    pi: u64 = 0,
   156	    pm: u64 = 0,
   157	    pl: u64 = 0,
   158	    pacc: u64 = 0,
   159	    // host tables
   160	    h_ids: [MAX_M]i32 = undefined,
   161	    h_bpos: [MAX_M]i32 = undefined,
   162	    h_rings: [MAX_S]u64 = undefined,
   163	    h_starts: [MAX_S]i64 = undefined,
   164	    h_froms: [MAX_S]i64 = undefined,
   165	    h_cpos: []i64 = &.{},
   166	    h_crings: []u64 = &.{},
   167	    cand: []i32 = &.{}, // all ranks' gathered candidates, [world][MAX_M][CW]
   168	    gh: []f32 = &.{}, // the selector's hidden projections [MAX_M][RANK]
   169	
   170	    pub fn deinit(dr: *Drafter) void {
   171	        for (dr.slots.items) |*s| if (s.ring) |*r| r.free();
   172	        dr.slots.deinit(dr.gpa);
   173	        dr.scratch.free();
   174	        dr.weights.free();
   175	        dr.module.unload();
   176	        const g = dr.gpa;
   177	        for ([_][]u16{ dr.pred, dr.succ }) |b| g.free(b);
   178	        g.free(dr.h_cpos);
   179	        g.free(dr.h_crings);
   180	        g.free(dr.cand);
   181	        g.free(dr.gh);
   182	        g.destroy(dr);
   183	    }
   184	
   185	    /// TF_GLM_DFLASH2_PROF: the stream idle, the time since `since` added to its phase; the new mark.
   186	    fn mark(dr: *Drafter, e: anytype, since: ?struct { i96, usize }) !i96 {
   187	        const io = dr.prof orelse return 0;
   188	        try e.stream.synchronize();
   189	        const now = std.Io.Clock.awake.now(io).toNanoseconds();
   190	        if (since) |s| dr.phase_ms[s[1]] += @as(f64, @floatFromInt(now - s[0])) / 1e6;
   191	        return now;
   192	    }
   193	
   194	    /// Where a forward's taps go: a prompt chunk's rows or a verify's.
   195	    pub fn taps(dr: *const Drafter, chunk: bool) taps_mod.Taps {
   196	        return .{ .f = dr.k.tap, .out = if (chunk) dr.taps_prefill else dr.taps_step };
   197	    }
   198	
   199	    fn slot(dr: *Drafter, i: u32) !*Slot {
   200	        if (i >= dr.slots.items.len) try dr.slots.appendNTimes(dr.gpa, .{}, i + 1 - dr.slots.items.len);
   201	        const s = &dr.slots.items[i];
   202	        if (s.ring == null) s.ring = try cuda.DeviceBuffer.alloc(dr.d, ring_bytes);
   203	        return s;
   204	    }
   205	
   206	    /// A fresh sequence on slot `i` whose rows are tapped from position `from` (0, or a kept prompt state's).
   207	    pub fn restart(dr: *Drafter, i: u32, from: usize) !void {
   208	        const s = try dr.slot(i);
   209	        s.from = @intCast(from);
   210	        s.first = no_rows;
   211	        s.end = @intCast(from);
   212	        s.v_rows = 0;
   213	    }
   214	
   215	    /// Slot `i`'s ring (made on first use), for a kept state to copy out of or into.
   216	    pub fn ringPtr(dr: *Drafter, i: u32) !u64 {
   217	        return (try dr.slot(i)).ring.?.ptr;
   218	    }
   219	
   220	    /// Where the context rows a state kept at `at` carries start: the prompt rows in the ring right up to `at`, at most
   221	    /// a window of them (null: the pass tapped none just before `at`, so the ring holds nothing a resume could use).
   222	    pub fn keptFrom(dr: *const Drafter, i: u32, at: usize) ?i64 {
   223	        if (i >= dr.slots.items.len) return null;
   224	        const s = &dr.slots.items[i];
   225	        const end: i64 = @intCast(at);
   226	        if (s.ring == null or s.first == no_rows or s.end != end or s.first >= end) return null;
   227	        return @max(@max(s.first, s.from), end - WIN);
   228	    }
   229	
   230	    /// A kept state's ring copied into slot `i`: its sequence's rows `from` .. `at` are in it, as a fresh pass left them.
   231	    pub fn resumeAt(dr: *Drafter, i: u32, from: i64, at: usize) !void {
   232	        const s = try dr.slot(i);
   233	        s.from = from;
   234	        s.first = from;
   235	        s.end = @intCast(at);
   236	        s.v_rows = 0;
   237	    }
   238	
   239	    /// A verify on slot `i` wrote `rows` tap rows at row `row0` of taps_step, for positions pos0...
   240	    pub fn noteVerify(dr: *Drafter, i: u32, row0: usize, rows: usize, pos0: usize, first: bool) !void {
   241	        if (first) dr.verifies += 1; // a shared verify's parts share one
   242	        const s = try dr.slot(i);
   243	        s.* = .{ .ring = s.ring, .from = s.from, .first = s.first, .end = s.end, .v_step = dr.verifies, .v_row0 = row0, .v_rows = rows, .v_pos0 = @intCast(pos0) };
   244	    }
   245	
   246	    /// TF_GLM_DFLASH2_RING_DIGEST=1 (a check, off by default): once a prompt is in, slot `i`'s whole context ring hashed
   247	    /// and logged for prompts of 8192 ids or more, so two runs (paired or a chunk at a time) can show the same context.
   248	    pub fn ringDigest(dr: *Drafter, e: anytype, i: u32, len: usize) !void {
   249	        const v = std.c.getenv("TF_GLM_DFLASH2_RING_DIGEST") orelse return;
   250	        if (!std.mem.eql(u8, std.mem.span(v), "1") or len < 8192) return; // long prompts only: a short one's time stays clean
   251	        const s = try dr.slot(i);
   252	        try e.stream.synchronize();
   253	        const host = try dr.gpa.alloc(u8, ring_bytes);
   254	        defer dr.gpa.free(host);
   255	        try s.ring.?.download(0, host);
   256	        std.log.info("dflash2: ring digest slot {d} after a {d}-id prompt: {x:0>16}", .{ i, len, std.hash.Wyhash.hash(0, host) });
   257	    }
   258	
   259	    /// A prompt chunk on slot `i` (rows start.. of a `len`-id prompt): its rows within the last window into the ring.
   260	    pub fn prompt(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize) !void {
   261	        return dr.promptSegment(e, i, start, rows, len, 0);
   262	    }
   263	
   264	    /// The same ordinary context ingest, with this prompt's rows at row0 in the batched tap buffer.
   265	    pub const promptSegment = ingestPrompt;
   266	
   267	    /// A paired step's second chunk whose tap rows are already in the context input from row `row0` (pairTaps).
   268	    pub const promptIn = ingestPromptIn;
   269	
   270	    /// Where a paired step's second chunk taps when the first's last `first_rows` rows go in too (cuda_prompt_pair):
   271	    /// the context input past them, idle through the pass. The first's ingest copies its rows in before them, then the
   272	    /// second's reads its own in place (promptIn), so a pair needs no tap rows of its own.
   273	    pub fn pairTaps(dr: *const Drafter, first_rows: usize) taps_mod.Taps {
   274	        return .{ .f = dr.k.tap, .out = dr.cin + first_rows * TAP_ROW };
   275	    }
   276	
   277	    /// The runs' rows through fc, the hidden norm and each layer's k, v into their rings.
   278	    pub fn ingest(dr: *Drafter, e: anytype, runs: []const Run) !void {
   279	        var N: usize = 0;
   280	        for (runs) |r| N += r.n;
   281	        if (N == 0) return;
   282	        if (N > MAX_CTX) return error.DrafterContextTooWide;
   283	        var row: usize = 0;
   284	        for (runs) |r| {
   285	            if (r.taps != dr.cin + row * TAP_ROW) try block.copy(e, dr.cin + row * TAP_ROW, r.taps, r.n * TAP_ROW); // in place: promptIn
   286	            for (0..r.n) |j| {
   287	                dr.h_cpos[row + j] = r.pos0 + @as(i64, @intCast(j));
   288	                dr.h_crings[row + j] = r.ring;
   289	            }
   290	            row += r.n;
   291	        }
   292	        try block.upload(dr, e, dr.cpos, std.mem.sliceAsBytes(dr.h_cpos[0..N]));
   293	        try block.upload(dr, e, dr.crings, std.mem.sliceAsBytes(dr.h_crings[0..N]));
   294	        try block.ingestPass(dr, e, N);
   295	    }
   296	
   297	    /// Kept rows into each context and no block pass (a copy takes these streams' next windows): every rank.
   298	    pub fn ingestOnly(dr: *Drafter, e: anytype, reqs: []const Req) !void {
   299	        if (reqs.len > MAX_S) return error.TooManyStreams;
   300	        var runs: [MAX_S]Run = undefined;
   301	        try dr.ingest(e, runs[0..try dr.keptRuns(reqs, &runs)]);
   302	    }
   303	
   304	    /// Each request's kept verify rows as a context run (a context whose rows are gone restarts at its position).
   305	    fn keptRuns(dr: *Drafter, reqs: []const Req, runs: *[MAX_S]Run) !usize {
   306	        var nr: usize = 0;
   307	        for (reqs) |q| {
   308	            const s = try dr.slot(q.slot);
   309	            if (q.n == 0) continue;
   310	            if (s.v_step != dr.verifies or q.n > s.v_rows or s.v_pos0 + @as(i64, @intCast(q.n)) != q.pos) {
   311	                s.from = q.pos; // its verify rows were overwritten (or never written): its context restarts here
   312	                dr.lost += 1;
   313	                continue;
   314	            }
   315	            runs[nr] = .{ .ring = s.ring.?.ptr, .taps = dr.taps_step + s.v_row0 * TAP_ROW, .n = q.n, .pos0 = s.v_pos0 };
   316	            nr += 1;
   317	        }
   318	        return nr;
   319	    }
   320	
   321	    /// Kept rows into each context, then the block pass for every stream (every rank); rank 0 selects 7 drafts each.
   322	    pub fn draft(dr: *Drafter, e: anytype, reqs: []const Req) !void {
   323	        if (reqs.len == 0) return;
   324	        if (reqs.len > MAX_S) return error.TooManyStreams;
   325	        var tm = try dr.mark(e, null);
   326	        var runs: [MAX_S]Run = undefined;
   327	        try dr.ingest(e, runs[0..try dr.keptRuns(reqs, &runs)]);
   328	        tm = try dr.mark(e, .{ tm, 0 });
   329	        const S = reqs.len;
   330	        for (reqs, 0..) |q, i| {
   331	            const s = &dr.slots.items[q.slot];
   332	            dr.h_rings[i] = s.ring.?.ptr;
   333	            dr.h_starts[i] = q.pos;
   334	            dr.h_froms[i] = s.from;
   335	            for (0..BLOCK) |t| {
   336	                dr.h_ids[i * BLOCK + t] = @intCast(if (t == 0) q.anchor else MASK);
   337	                dr.h_bpos[i * BLOCK + t] = @intCast(q.pos + @as(i64, @intCast(t)));
   338	            }
   339	        }
   340	        const M = S * BLOCK;
   341	        try block.upload(dr, e, dr.ids, std.mem.sliceAsBytes(dr.h_ids[0..M]));
   342	        try block.upload(dr, e, dr.bpos, std.mem.sliceAsBytes(dr.h_bpos[0..M]));
   343	        try block.upload(dr, e, dr.rings, std.mem.sliceAsBytes(dr.h_rings[0..S]));
   344	        try block.upload(dr, e, dr.starts, std.mem.sliceAsBytes(dr.h_starts[0..S]));
   345	        try block.upload(dr, e, dr.froms, std.mem.sliceAsBytes(dr.h_froms[0..S]));
   346	        const rank0 = reqs[0].out != null;
   347	        tm = try dr.mark(e, .{ tm, 1 });
   348	        try block.pass(dr, e, S, rank0);
   349	        tm = try dr.mark(e, .{ tm, 2 });
   350	        if (rank0) {
   351	            try block.download(dr, e, std.mem.sliceAsBytes(dr.cand[0..@import("dflash_candidates.zig").Layout.init(dr.world, M, CW).read]), dr.gtk);
   352	            try block.download(dr, e, std.mem.sliceAsBytes(dr.gh[0 .. M * RANK]), dr.g);
   353	            for (reqs, 0..) |q, i| select.select(dr, i, M, q);
   354	            if (dr.log > dr.calls) for (reqs) |q| std.log.info("dflash2 round {d}: slot {d} pos {d} kept {d} anchor {d} drafts {any}", .{ dr.calls, q.slot, q.pos, q.n, q.anchor, q.out.?[0 .. BLOCK - 1] });
   355	            if (dr.calls % 256 == 255) std.log.info("dflash2: {d} rounds, {d} contexts restarted", .{ dr.calls + 1, dr.lost });
   356	        }
   357	        _ = try dr.mark(e, .{ tm, 3 });
   358	        dr.calls += 1;
   359	        if (dr.prof != null and dr.calls % 64 == 0) {
   360	            const k: f64 = 64;
   361	            std.log.info("dflash2 ms a round: ingest {d:.2} tables {d:.2} block {d:.2} select {d:.2}", .{ dr.phase_ms[0] / k, dr.phase_ms[1] / k, dr.phase_ms[2] / k, dr.phase_ms[3] / k });
   362	            dr.phase_ms = @splat(0);
   363	        }
   364	    }
   365	};
   366	
   367	/// Prompt context retained in the window; row0 is only a tap-buffer offset, never a sequence position.
   368	pub fn promptRun(ring: u64, taps: u64, start: usize, rows: usize, len: usize, row0: usize) !?Run {
   369	    if (start > len or rows > len - start) return error.BadPromptContext;
   370	    if (rows > MAX_CTX or row0 > MAX_CTX - rows) return error.DrafterContextTooWide;
   371	    const keep_from = len -| @as(usize, @intCast(WIN));
   372	    if (start + rows <= keep_from) return null;
   373	    const lo = keep_from -| start;
   374	    return .{ .ring = ring, .taps = taps + (row0 + lo) * TAP_ROW, .n = rows - lo, .pos0 = @intCast(start + lo) };
   375	}
   376	
   377	pub fn ingestPrompt(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, row0: usize) !void {
   378	    return ingestFrom(dr, e, i, start, rows, len, dr.taps_prefill, row0);
   379	}
   380	
   381	/// ingestPrompt of a chunk whose tap rows already sit in the context input from row `row0` on (a paired step's second
   382	/// chunk, tapped at pairTaps while the first's rows go in before them): the ingest reads them where they are, as the
   383	/// copy would have left them at row 0. Every one of its rows is in the last window.
   384	pub fn ingestPromptIn(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, row0: usize) !void {
   385	    if (start < len -| @as(usize, @intCast(WIN)) or row0 > MAX_CTX or rows > MAX_CTX - row0) return error.BadPromptContext;
   386	    const cin = dr.cin;
   387	    defer dr.cin = cin;
   388	    dr.cin = cin + row0 * TAP_ROW;
   389	    return ingestFrom(dr, e, i, start, rows, len, dr.cin, 0);
   390	}
   391	
   392	/// A prompt chunk's rows in the last window from tap rows at `taps` (row `row0` on) into slot `i`'s ring, and the run
   393	/// the ring then holds (keptFrom's), whichever buffer they were tapped into.
   394	fn ingestFrom(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, taps: u64, row0: usize) !void {
   395	    const s = try dr.slot(i);
   396	    const run = (try promptRun(s.ring.?.ptr, taps, start, rows, len, row0)) orelse return;
   397	    try dr.ingest(e, &.{run});
   398	    noteRun(s, run);
   399	}
   400	
   401	/// The run slot `s`'s ring holds once `run` is in: it goes on from `end`, or starts again past a gap (rows a kept
   402	/// state may carry).
   403	pub fn noteRun(s: anytype, run: Run) void {
   404	    if (s.first == no_rows or run.pos0 != s.end) s.first = run.pos0;
   405	    s.end = run.pos0 + @as(i64, @intCast(run.n));
   406	}

exec
/bin/bash -lc 'nl -ba src/cuda_prompt_pair.zig' in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
     1	//! A prompt step of two chunks in one pass (TF_GLM_PROMPT_PAIR): each chunk's exchanges under the other's sites.
     2	const std = @import("std");
     3	const cuda = @import("cuda");
     4	const fw = @import("cuda_forward.zig");
     5	const weights = @import("cuda_weights.zig");
     6	const snapshots = @import("cuda_snapshots.zig");
     7	const engine = @import("cuda_engine.zig");
     8	const dflash = @import("dflash.zig");
     9	const Engine = engine.Engine;
    10	
    11	/// The fewest rows a paired step's second chunk takes: a shorter one's sites end before the first's exchanges do.
    12	pub const min_rows = fw.prefill_rows / 4;
    13	
    14	/// The shortest prompt (ids) whose steps pair: three chunks or more by default. A prompt of two chunks is one paired
    15	/// step with nothing after it to hide, and measured ~3% slower paired at 4K (2x RTX PRO 6000, DFlash2); from 16K on a
    16	/// pair is ahead. TF_GLM_PROMPT_PAIR_MIN overrides it in ids (0: pair wherever a step can). Rank 0's choice, so the
    17	/// other ranks need no setting: they take the steps it sends.
    18	pub const default_min_len = 2 * fw.prefill_rows + 1;
    19	
    20	pub fn minLen() usize {
    21	    const v = std.c.getenv("TF_GLM_PROMPT_PAIR_MIN") orelse return default_min_len;
    22	    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default_min_len;
    23	}
    24	
    25	/// A paired step's second chunk: twins of the prompt buffers, pieces on their side stream, and its first position.
    26	/// Beside DFlash2 it taps into the drafter's own buffers (dflash pairTaps), so it holds no tap rows.
    27	pub const Pair = struct {
    28	    bp: fw.Buffers,
    29	    pieces: fw.Pieces,
    30	    pos: cuda.DeviceBuffer, // an i32: where the second chunk's view of the sequence starts
    31	    own_room: bool, // its own exchange room; else the first chunk's (split exchanges: only the side stream touches it)
    32	    steps: u64 = 0, // paired steps so far (glm-pair-check counts them)
    33	
    34	    /// `own_room`: the second chunk exchanges into rooms of its own (TF_GLM_PREFILL_REDUCE=gather fills a room on the
    35	    /// main stream); split exchanges share the first's (64 MiB a rank at TP=2).
    36	    pub fn init(d: *const cuda.Driver, w: *const weights.Weights, bp: *const fw.Buffers, pieces: *const fw.Pieces, capacity: usize, own_room: bool) !Pair {
    37	        var b = try fw.Buffers.initTwin(d, w, bp, capacity, own_room);
    38	        errdefer b.deinit();
    39	        var p = try fw.Pieces.initTwin(d, pieces);
    40	        errdefer p.deinit();
    41	        return .{ .bp = b, .pieces = p, .pos = try cuda.DeviceBuffer.alloc(d, 4), .own_room = own_room };
    42	    }
    43	
    44	    pub fn deinit(p: *Pair) void {
    45	        p.pos.free();
    46	        p.pieces.deinit();
    47	        p.bp.deinit();
    48	        p.* = undefined;
    49	    }
    50	};
    51	
    52	/// Whether an engine makes the pair's twin: TF_GLM_PROMPT_PAIR on, and other ranks whose exchanges a pair hides.
    53	pub fn wanted(on: bool, world: usize) bool {
    54	    return on and world > 1;
    55	}
    56	
    57	/// Whether a step at `start` of a `len`-id prompt takes the next chunk too: min_rows or more there, and no mark at it.
    58	pub fn takesTwo(start: usize, len: usize, world: usize, marks: []const u32) bool {
    59	    const next = start + fw.prefill_rows;
    60	    if (next >= len or len - next < @max(min_rows, world)) return false;
    61	    return std.mem.indexOfScalar(u32, marks, @intCast(next)) == null;
    62	}
    63	
    64	/// Whether chunks of `ra` and `rb` rows can share one exchange room (Pair.own_room false): every piece of each splits
    65	/// (at least a row a rank), so only the side stream, which posts both chunks' exchanges in one order, touches it.
    66	pub fn piecesSplit(ra: usize, rb: usize, count: usize, world: usize) bool {
    67	    var cut: [fw.max_pieces][2]usize = undefined;
    68	    for ([_]usize{ ra, rb }) |rows| {
    69	        const n = fw.ranges(rows, count, &cut);
    70	        for (cut[0..n]) |p| if (p[1] - p[0] < world) return false;
    71	    }
    72	    return true;
    73	}
    74	
    75	/// Whether a pair at `start` (b of `rb` rows) rebases the index rings where chunks one at a time do: at a, or not.
    76	pub fn ringsAgree(base: usize, ring: usize, start: usize, rb: usize) bool {
    77	    const ra = fw.prefill_rows;
    78	    return start + ra > base + ring or start + ra + rb <= base + ring;
    79	}
    80	
    81	/// Rank 0's choice for the step at `start` (the others take the ids it sends); not with prompt rows or prof.
    82	pub fn pairedAt(e: *const Engine, start: usize, len: usize, from: ?engine.Resume, marks: ?engine.Marks) bool {
    83	    if (e.pair == null or !e.settings.pair or e.prompt_logprobs != null or e.prompt_profs[3] != null) return false;
    84	    if (e.pieces.count == 1 and !e.settings.split) return false; // run takes such a chunk whole
    85	    if (len < minLen()) return false; // too short to gain from a pair
    86	    if (!takesTwo(start, len, e.share.world, if (marks) |m| m.at else &.{})) return false;
    87	    const rb = @min(len - start - fw.prefill_rows, fw.prefill_rows);
    88	    if (!e.pair.?.own_room and !piecesSplit(fw.prefill_rows, rb, e.pieces.count, e.share.world)) return false;
    89	    // a resumed step restores its kept state, and the rings' base, after this choice
    90	    const base = if (from) |r| (e.snaps.get(r.id) orelse return false).bases[0] else e.cur.st.base;
    91	    return ringsAgree(base, e.cur.st.ring_rows, start, rb);
    92	}
    93	
    94	/// Engine.prefillChunk of two chunks (`part`: both, then the next id): one pass, then each chunk's MTP rows and commit.
    95	/// With DFlash2 each chunk taps where prefillChunk would (a chunk reaching the last window), and the drafter ingests
    96	/// a's rows, then b's, as two chunks one at a time do.
    97	pub fn prefill(e: *Engine, part: []const u32, start: usize, len: usize, cancel: ?engine.Cancel, marks: ?engine.Marks, split: bool) !?u32 {
    98	    const ra: usize = fw.prefill_rows;
    99	    if (len <= start + ra) return error.BadChunk;
   100	    const rb: usize = @min(len - start - ra, fw.prefill_rows);
   101	    const pair = if (e.pair) |*p| p else return error.BadChunk;
   102	    if (rb < e.share.world or part.len != @min(len - start, ra + rb + 1) or e.prompt_logprobs != null) return error.BadChunk;
   103	    if (!pair.own_room and !piecesSplit(ra, rb, e.pieces.count, e.share.world)) return error.BadChunk; // pairedAt's rule
   104	    if (!ringsAgree(e.cur.st.base, e.cur.st.ring_rows, start, rb)) return error.BadChunk; // pairedAt's rule, every rank
   105	    if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
   106	    e.gate(cancel) catch |err| {
   107	        if (err == error.Cancelled) try snapshots.interrupted(e, marks, @intCast(start));
   108	        return err;
   109	    };
   110	    var one: [1]fw.Rank = undefined;
   111	    var two: [1]fw.Rank = undefined;
   112	    var fa = e.view(&e.bp, &one);
   113	    var fb = e.view(&pair.bp, &two);
   114	    fa.prof = null;
   115	    fb.prof = null;
   116	    fa.index_split = split; // the step's choice (TF_GLM_INDEX_SPLIT) for both chunks
   117	    fb.index_split = split;
   118	    // DFlash2 taps (prefillChunk's rule): only a chunk that reaches into the last window. a's go to taps_prefill; b's
   119	    // there too when a has none, else into the drafter's context input past a's kept rows (dflash pairTaps)
   120	    const keep_from = len -| @as(usize, @intCast(dflash.WIN));
   121	    const a_kept = start + ra -| keep_from; // a's rows in the last window (keep_from > start: b follows a)
   122	    if (a_kept == 0) fa.taps = null;
   123	    fb.taps = if (e.drafter) |dr| (if (start + ra + rb <= keep_from) null else if (a_kept > 0) dr.pairTaps(a_kept) else dr.taps(true)) else null;
   124	    const st = &e.cur.st;
   125	    try st.ready(e.stream, ra + rb); // one rebase for both chunks, at a as a chunk at a time, before b's view copies it
   126	    var sb = st.*;
   127	    sb.pos_dev = pair.pos.ptr;
   128	    try sb.setPos(e.stream, st.pos + ra);
   129	    two[0].st = &sb;
   130	    st.pending = ra;
   131	    st.prompt = true;
   132	    const ids: []const i32 = @ptrCast(part);
   133	    try cuda.copy_source.enqueue(e.d, e.bp.ids, std.mem.sliceAsBytes(ids[0..ra]), e.stream);
   134	    try cuda.copy_source.enqueue(e.d, pair.bp.ids, std.mem.sliceAsBytes(ids[ra .. ra + rb]), e.stream);
   135	    try fw.Forward.pairBody(&fa, &fb, ra, rb);
   136	    pair.steps += 1;
   137	    e.index_splits += @intFromBool(split);
   138	    if (e.drafter) |dr| { // a's rows from taps_prefill, then b's where they were tapped, each as its own chunk's would
   139	        try dr.prompt(e, e.slot, start, ra, len);
   140	        if (a_kept > 0) try dr.promptIn(e, e.slot, start + ra, rb, len, a_kept) else try dr.prompt(e, e.slot, start + ra, rb, len);
   141	        if (start + ra + rb == len) try dr.ringDigest(e, e.slot, len);
   142	    }
   143	    try mtpRows(e, &fa, ids[1 .. ra + 1], e.bp.fnormed);
   144	    try fa.commit(ra);
   145	    st.pending = rb;
   146	    st.prompt = true;
   147	    var first: ?u32 = null;
   148	    if (start + ra + rb == len) {
   149	        var t: [1]u32 = undefined;
   150	        try e.draw(pair.bp.logits, 1, len, 0, &t);
   151	        first = t[0];
   152	        try e.holdPrompt(pair.bp.fnormed + (rb - 1) * e.c.hidden * 2);
   153	    }
   154	    try mtpRows(e, &fa, ids[ra + 1 ..], pair.bp.fnormed); // the MTP head on a's buffers, over b's rows
   155	    try fa.commit(rb);
   156	    return first;
   157	}
   158	
   159	/// decode._absorb_rows of a chunk: the MTP head over its next ids `nxt` on its final-norm rows `hidden`.
   160	fn mtpRows(e: *Engine, f: *fw.Forward, nxt: []const i32, hidden: u64) !void {
   161	    if (!e.mtp or e.drafter != null or nxt.len == 0) return; // beside a drafter the head's caches are never read
   162	    try f.mtp(nxt, &.{hidden});
   163	    try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + nxt.len);
   164	}
   165	
   166	test "a prompt step takes the next chunk when it holds min_rows or more and no kept state starts there" {
   167	    const C = fw.prefill_rows;
   168	    try std.testing.expect(takesTwo(0, 2 * C, 3, &.{}));
   169	    try std.testing.expect(takesTwo(0, C + min_rows, 3, &.{}));
   170	    try std.testing.expect(!takesTwo(0, C + min_rows - 1, 3, &.{})); // a shorter tail goes alone
   171	    try std.testing.expect(!takesTwo(0, C, 3, &.{}));
   172	    try std.testing.expect(!takesTwo(C, 2 * C, 3, &.{})); // the prompt's last chunk
   173	    try std.testing.expect(takesTwo(2 * C, 5 * C, 3, &.{}));
   174	    try std.testing.expect(!takesTwo(0, 3 * C, 3, &.{C})); // a state kept at the next chunk's start: that chunk starts a step
   175	    try std.testing.expect(takesTwo(0, 3 * C, 3, &.{ 2 * C, 3 * C })); // marks past the pair
   176	}
   177	
   178	test "a pair rebases the index rings where a chunk at a time would, so a ring based at chunk 0 pairs from chunk 2" {
   179	    const C = fw.prefill_rows;
   180	    try std.testing.expect(ringsAgree(0, 2 * C, 0, C)); // neither rebases
   181	    try std.testing.expect(!ringsAgree(0, 2 * C, C, C)); // a chunk at a time rebases at b's start, a pair at a's
   182	    try std.testing.expect(!ringsAgree(0, 2 * C, C, 1));
   183	    try std.testing.expect(ringsAgree(0, 2 * C, 2 * C, C)); // both at a's start
   184	    try std.testing.expect(ringsAgree(C, 2 * C, C, C));
   185	}
   186	
   187	test "chunks share one exchange room only where every piece splits: a row a rank in each" {
   188	    const C = fw.prefill_rows;
   189	    try std.testing.expect(piecesSplit(C, C, 3, 2));
   190	    try std.testing.expect(piecesSplit(C, min_rows, 3, 2));
   191	    try std.testing.expect(!piecesSplit(C, 769, 3, 2)); // pieces of 384, 384 and 1 row: the last goes whole
   192	    try std.testing.expect(piecesSplit(C, 770, 3, 2));
   193	    try std.testing.expect(!piecesSplit(C, 770, 3, 3));
   194	    try std.testing.expect(piecesSplit(C, 769, 1, 2)); // one piece of every row
   195	    for (1..fw.max_pieces + 1) |count| try std.testing.expect(piecesSplit(C, C, count, 16)); // a's pieces always split
   196	}
   197	
   198	test "only ranks that exchange make the twin: TP=1 has no exchanges to hide" {
   199	    try std.testing.expect(!wanted(true, 1));
   200	    try std.testing.expect(wanted(true, 2));
   201	    try std.testing.expect(!wanted(false, 3));
   202	}

exec
/bin/bash -lc "rg -n 'prompt_pair|ingest|DrafterContextTooWide|ctx_start|\\.gs\\b|\\.taps|ring_bytes|rope|rotary' src/dflash* src/cuda_prompt* src/cuda* | head -n 250" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
src/cuda_prompt_pair.zig:95:/// With DFlash2 each chunk taps where prefillChunk would (a chunk reaching the last window), and the drafter ingests
src/cuda_prompt_pair.zig:122:    if (a_kept == 0) fa.taps = null;
src/cuda_prompt_pair.zig:123:    fb.taps = if (e.drafter) |dr| (if (start + ra + rb <= keep_from) null else if (a_kept > 0) dr.pairTaps(a_kept) else dr.taps(true)) else null;
src/cuda_forward.zig:416:            if (f.taps) |t| try f.tapPieces(t, j, cut);
src/cuda_forward.zig:422:        if (f.taps) |t| try f.shareTaps(t, cut);
src/cuda_forward.zig:478:            if (f.taps) |t| try t.at(f.stream, j, f.ranks[0].b.x, 0, R);
src/cuda_prompt_pair.zig:95:/// With DFlash2 each chunk taps where prefillChunk would (a chunk reaching the last window), and the drafter ingests
src/cuda_prompt_pair.zig:122:    if (a_kept == 0) fa.taps = null;
src/cuda_prompt_pair.zig:123:    fb.taps = if (e.drafter) |dr| (if (start + ra + rb <= keep_from) null else if (a_kept > 0) dr.pairTaps(a_kept) else dr.taps(true)) else null;
src/cuda_engine.zig:21:const prompt_pair = @import("cuda_prompt_pair.zig");
src/cuda_engine.zig:103:    pair: ?prompt_pair.Pair = null, // TF_GLM_PROMPT_PAIR: a paired prompt step's second chunk
src/cuda_engine.zig:175:        const pair = prompt_pair.wanted(o.settings.pair, world);
src/cuda_engine.zig:225:        if (pair) e.pair = try prompt_pair.Pair.init(d, &e.w, &e.bp, &e.pieces, o.capacity, !o.settings.split); // split: one room
src/cuda_engine.zig:306:            e.m.mm.a8 = .{ .q = buf.ptr, .xs = buf.ptr + sz[0], .rows = fw.prefill_rows, .k = a8_max_k, .gs = gs };
src/cuda_engine.zig:319:            if (e.m.mm.a8) |a| (if (a.gs == 32) "int8, a scale per 32 (TF_GLM_W8A8=32)" else "int8, a scale per 64 (TF_GLM_W8A8=1)") else "bf16",
src/cuda_engine.zig:331:        return .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = out, .exchange = e.exchange(), .pieces = if (b == &e.bp) &e.pieces else twin, .split = e.settings.split, .ring = e.settings.ring, .kda_wide = e.settings.kda_wide, .absorb_mma = e.settings.numerics.absorb_mma, .images = im, .taps = if (e.drafter) |dr| dr.taps(b == &e.bp) else null, .side = if (b.prefill) null else e.side, .moe_fuse = e.moe_fuse, .ce = if (b == &e.bp) e.ce else null }; // prompt buffers (the pair's twin too) keep the shared expert on their stream; the twin's exchanges stay on NCCL (its rows are not registered)
src/cuda_engine.zig:451:        if (part.len > rows + 1) return prompt_pair.prefill(e, part, start, len, cancel, marks, split);
src/cuda_engine.zig:456:        if (start + rows <= len -| @as(usize, @intCast(dflash.WIN))) f.taps = null; // none read (dflash.promptRun)
src/cuda_engine.zig:620:        const f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = null, .split = e.settings.split, .ring = e.settings.ring, .kda_wide = e.settings.kda_wide, .absorb_mma = e.settings.numerics.absorb_mma, .taps = if (e.drafter) |dr| dr.taps(false) else null, .side = e.side, .moe_fuse = e.moe_fuse };
src/cuda_engine.zig:638:    pub fn ingestBlock(e: *Engine, reqs: []dflash.Req) !void {
src/cuda_engine.zig:641:        try dr.ingestOnly(e, reqs);
src/cuda_engine.zig:645:    pub const pairedAt = prompt_pair.pairedAt;
src/dflash.zig:30:pub const MAX_CTX: usize = 2048; // context rows one ingest takes (a prompt chunk's)
src/dflash.zig:35:pub const ring_bytes: u64 = LAYERS * 2 * @as(u64, @intCast(WIN)) * KVW * 2;
src/dflash.zig:92:/// Every rank runs the ingest and the block pass on its half of the MLP and head; rank 0 selects the drafts.
src/dflash.zig:118:    phase_ms: [4]f64 = @splat(0), // ingest, tables, block pass, selector
src/dflash.zig:196:        return .{ .f = dr.k.tap, .out = if (chunk) dr.taps_prefill else dr.taps_step };
src/dflash.zig:202:        if (s.ring == null) s.ring = try cuda.DeviceBuffer.alloc(dr.d, ring_bytes);
src/dflash.zig:253:        const host = try dr.gpa.alloc(u8, ring_bytes);
src/dflash.zig:264:    /// The same ordinary context ingest, with this prompt's rows at row0 in the batched tap buffer.
src/dflash.zig:265:    pub const promptSegment = ingestPrompt;
src/dflash.zig:268:    pub const promptIn = ingestPromptIn;
src/dflash.zig:270:    /// Where a paired step's second chunk taps when the first's last `first_rows` rows go in too (cuda_prompt_pair):
src/dflash.zig:271:    /// the context input past them, idle through the pass. The first's ingest copies its rows in before them, then the
src/dflash.zig:278:    pub fn ingest(dr: *Drafter, e: anytype, runs: []const Run) !void {
src/dflash.zig:282:        if (N > MAX_CTX) return error.DrafterContextTooWide;
src/dflash.zig:285:            if (r.taps != dr.cin + row * TAP_ROW) try block.copy(e, dr.cin + row * TAP_ROW, r.taps, r.n * TAP_ROW); // in place: promptIn
src/dflash.zig:294:        try block.ingestPass(dr, e, N);
src/dflash.zig:298:    pub fn ingestOnly(dr: *Drafter, e: anytype, reqs: []const Req) !void {
src/dflash.zig:301:        try dr.ingest(e, runs[0..try dr.keptRuns(reqs, &runs)]);
src/dflash.zig:315:            runs[nr] = .{ .ring = s.ring.?.ptr, .taps = dr.taps_step + s.v_row0 * TAP_ROW, .n = q.n, .pos0 = s.v_pos0 };
src/dflash.zig:327:        try dr.ingest(e, runs[0..try dr.keptRuns(reqs, &runs)]);
src/dflash.zig:361:            std.log.info("dflash2 ms a round: ingest {d:.2} tables {d:.2} block {d:.2} select {d:.2}", .{ dr.phase_ms[0] / k, dr.phase_ms[1] / k, dr.phase_ms[2] / k, dr.phase_ms[3] / k });
src/dflash.zig:370:    if (rows > MAX_CTX or row0 > MAX_CTX - rows) return error.DrafterContextTooWide;
src/dflash.zig:374:    return .{ .ring = ring, .taps = taps + (row0 + lo) * TAP_ROW, .n = rows - lo, .pos0 = @intCast(start + lo) };
src/dflash.zig:377:pub fn ingestPrompt(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, row0: usize) !void {
src/dflash.zig:378:    return ingestFrom(dr, e, i, start, rows, len, dr.taps_prefill, row0);
src/dflash.zig:381:/// ingestPrompt of a chunk whose tap rows already sit in the context input from row `row0` on (a paired step's second
src/dflash.zig:382:/// chunk, tapped at pairTaps while the first's rows go in before them): the ingest reads them where they are, as the
src/dflash.zig:384:pub fn ingestPromptIn(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, row0: usize) !void {
src/dflash.zig:389:    return ingestFrom(dr, e, i, start, rows, len, dr.cin, 0);
src/dflash.zig:394:fn ingestFrom(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, taps: u64, row0: usize) !void {
src/dflash.zig:397:    try dr.ingest(e, &.{run});

exec
/bin/bash -lc "nl -ba src/dflash.zig | sed -n '175,295p;365,430p'" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
   175	        dr.module.unload();
   176	        const g = dr.gpa;
   177	        for ([_][]u16{ dr.pred, dr.succ }) |b| g.free(b);
   178	        g.free(dr.h_cpos);
   179	        g.free(dr.h_crings);
   180	        g.free(dr.cand);
   181	        g.free(dr.gh);
   182	        g.destroy(dr);
   183	    }
   184	
   185	    /// TF_GLM_DFLASH2_PROF: the stream idle, the time since `since` added to its phase; the new mark.
   186	    fn mark(dr: *Drafter, e: anytype, since: ?struct { i96, usize }) !i96 {
   187	        const io = dr.prof orelse return 0;
   188	        try e.stream.synchronize();
   189	        const now = std.Io.Clock.awake.now(io).toNanoseconds();
   190	        if (since) |s| dr.phase_ms[s[1]] += @as(f64, @floatFromInt(now - s[0])) / 1e6;
   191	        return now;
   192	    }
   193	
   194	    /// Where a forward's taps go: a prompt chunk's rows or a verify's.
   195	    pub fn taps(dr: *const Drafter, chunk: bool) taps_mod.Taps {
   196	        return .{ .f = dr.k.tap, .out = if (chunk) dr.taps_prefill else dr.taps_step };
   197	    }
   198	
   199	    fn slot(dr: *Drafter, i: u32) !*Slot {
   200	        if (i >= dr.slots.items.len) try dr.slots.appendNTimes(dr.gpa, .{}, i + 1 - dr.slots.items.len);
   201	        const s = &dr.slots.items[i];
   202	        if (s.ring == null) s.ring = try cuda.DeviceBuffer.alloc(dr.d, ring_bytes);
   203	        return s;
   204	    }
   205	
   206	    /// A fresh sequence on slot `i` whose rows are tapped from position `from` (0, or a kept prompt state's).
   207	    pub fn restart(dr: *Drafter, i: u32, from: usize) !void {
   208	        const s = try dr.slot(i);
   209	        s.from = @intCast(from);
   210	        s.first = no_rows;
   211	        s.end = @intCast(from);
   212	        s.v_rows = 0;
   213	    }
   214	
   215	    /// Slot `i`'s ring (made on first use), for a kept state to copy out of or into.
   216	    pub fn ringPtr(dr: *Drafter, i: u32) !u64 {
   217	        return (try dr.slot(i)).ring.?.ptr;
   218	    }
   219	
   220	    /// Where the context rows a state kept at `at` carries start: the prompt rows in the ring right up to `at`, at most
   221	    /// a window of them (null: the pass tapped none just before `at`, so the ring holds nothing a resume could use).
   222	    pub fn keptFrom(dr: *const Drafter, i: u32, at: usize) ?i64 {
   223	        if (i >= dr.slots.items.len) return null;
   224	        const s = &dr.slots.items[i];
   225	        const end: i64 = @intCast(at);
   226	        if (s.ring == null or s.first == no_rows or s.end != end or s.first >= end) return null;
   227	        return @max(@max(s.first, s.from), end - WIN);
   228	    }
   229	
   230	    /// A kept state's ring copied into slot `i`: its sequence's rows `from` .. `at` are in it, as a fresh pass left them.
   231	    pub fn resumeAt(dr: *Drafter, i: u32, from: i64, at: usize) !void {
   232	        const s = try dr.slot(i);
   233	        s.from = from;
   234	        s.first = from;
   235	        s.end = @intCast(at);
   236	        s.v_rows = 0;
   237	    }
   238	
   239	    /// A verify on slot `i` wrote `rows` tap rows at row `row0` of taps_step, for positions pos0...
   240	    pub fn noteVerify(dr: *Drafter, i: u32, row0: usize, rows: usize, pos0: usize, first: bool) !void {
   241	        if (first) dr.verifies += 1; // a shared verify's parts share one
   242	        const s = try dr.slot(i);
   243	        s.* = .{ .ring = s.ring, .from = s.from, .first = s.first, .end = s.end, .v_step = dr.verifies, .v_row0 = row0, .v_rows = rows, .v_pos0 = @intCast(pos0) };
   244	    }
   245	
   246	    /// TF_GLM_DFLASH2_RING_DIGEST=1 (a check, off by default): once a prompt is in, slot `i`'s whole context ring hashed
   247	    /// and logged for prompts of 8192 ids or more, so two runs (paired or a chunk at a time) can show the same context.
   248	    pub fn ringDigest(dr: *Drafter, e: anytype, i: u32, len: usize) !void {
   249	        const v = std.c.getenv("TF_GLM_DFLASH2_RING_DIGEST") orelse return;
   250	        if (!std.mem.eql(u8, std.mem.span(v), "1") or len < 8192) return; // long prompts only: a short one's time stays clean
   251	        const s = try dr.slot(i);
   252	        try e.stream.synchronize();
   253	        const host = try dr.gpa.alloc(u8, ring_bytes);
   254	        defer dr.gpa.free(host);
   255	        try s.ring.?.download(0, host);
   256	        std.log.info("dflash2: ring digest slot {d} after a {d}-id prompt: {x:0>16}", .{ i, len, std.hash.Wyhash.hash(0, host) });
   257	    }
   258	
   259	    /// A prompt chunk on slot `i` (rows start.. of a `len`-id prompt): its rows within the last window into the ring.
   260	    pub fn prompt(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize) !void {
   261	        return dr.promptSegment(e, i, start, rows, len, 0);
   262	    }
   263	
   264	    /// The same ordinary context ingest, with this prompt's rows at row0 in the batched tap buffer.
   265	    pub const promptSegment = ingestPrompt;
   266	
   267	    /// A paired step's second chunk whose tap rows are already in the context input from row `row0` (pairTaps).
   268	    pub const promptIn = ingestPromptIn;
   269	
   270	    /// Where a paired step's second chunk taps when the first's last `first_rows` rows go in too (cuda_prompt_pair):
   271	    /// the context input past them, idle through the pass. The first's ingest copies its rows in before them, then the
   272	    /// second's reads its own in place (promptIn), so a pair needs no tap rows of its own.
   273	    pub fn pairTaps(dr: *const Drafter, first_rows: usize) taps_mod.Taps {
   274	        return .{ .f = dr.k.tap, .out = dr.cin + first_rows * TAP_ROW };
   275	    }
   276	
   277	    /// The runs' rows through fc, the hidden norm and each layer's k, v into their rings.
   278	    pub fn ingest(dr: *Drafter, e: anytype, runs: []const Run) !void {
   279	        var N: usize = 0;
   280	        for (runs) |r| N += r.n;
   281	        if (N == 0) return;
   282	        if (N > MAX_CTX) return error.DrafterContextTooWide;
   283	        var row: usize = 0;
   284	        for (runs) |r| {
   285	            if (r.taps != dr.cin + row * TAP_ROW) try block.copy(e, dr.cin + row * TAP_ROW, r.taps, r.n * TAP_ROW); // in place: promptIn
   286	            for (0..r.n) |j| {
   287	                dr.h_cpos[row + j] = r.pos0 + @as(i64, @intCast(j));
   288	                dr.h_crings[row + j] = r.ring;
   289	            }
   290	            row += r.n;
   291	        }
   292	        try block.upload(dr, e, dr.cpos, std.mem.sliceAsBytes(dr.h_cpos[0..N]));
   293	        try block.upload(dr, e, dr.crings, std.mem.sliceAsBytes(dr.h_crings[0..N]));
   294	        try block.ingestPass(dr, e, N);
   295	    }
   365	};
   366	
   367	/// Prompt context retained in the window; row0 is only a tap-buffer offset, never a sequence position.
   368	pub fn promptRun(ring: u64, taps: u64, start: usize, rows: usize, len: usize, row0: usize) !?Run {
   369	    if (start > len or rows > len - start) return error.BadPromptContext;
   370	    if (rows > MAX_CTX or row0 > MAX_CTX - rows) return error.DrafterContextTooWide;
   371	    const keep_from = len -| @as(usize, @intCast(WIN));
   372	    if (start + rows <= keep_from) return null;
   373	    const lo = keep_from -| start;
   374	    return .{ .ring = ring, .taps = taps + (row0 + lo) * TAP_ROW, .n = rows - lo, .pos0 = @intCast(start + lo) };
   375	}
   376	
   377	pub fn ingestPrompt(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, row0: usize) !void {
   378	    return ingestFrom(dr, e, i, start, rows, len, dr.taps_prefill, row0);
   379	}
   380	
   381	/// ingestPrompt of a chunk whose tap rows already sit in the context input from row `row0` on (a paired step's second
   382	/// chunk, tapped at pairTaps while the first's rows go in before them): the ingest reads them where they are, as the
   383	/// copy would have left them at row 0. Every one of its rows is in the last window.
   384	pub fn ingestPromptIn(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, row0: usize) !void {
   385	    if (start < len -| @as(usize, @intCast(WIN)) or row0 > MAX_CTX or rows > MAX_CTX - row0) return error.BadPromptContext;
   386	    const cin = dr.cin;
   387	    defer dr.cin = cin;
   388	    dr.cin = cin + row0 * TAP_ROW;
   389	    return ingestFrom(dr, e, i, start, rows, len, dr.cin, 0);
   390	}
   391	
   392	/// A prompt chunk's rows in the last window from tap rows at `taps` (row `row0` on) into slot `i`'s ring, and the run
   393	/// the ring then holds (keptFrom's), whichever buffer they were tapped into.
   394	fn ingestFrom(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, taps: u64, row0: usize) !void {
   395	    const s = try dr.slot(i);
   396	    const run = (try promptRun(s.ring.?.ptr, taps, start, rows, len, row0)) orelse return;
   397	    try dr.ingest(e, &.{run});
   398	    noteRun(s, run);
   399	}
   400	
   401	/// The run slot `s`'s ring holds once `run` is in: it goes on from `end`, or starts again past a gap (rows a kept
   402	/// state may carry).
   403	pub fn noteRun(s: anytype, run: Run) void {
   404	    if (s.first == no_rows or run.pos0 != s.end) s.first = run.pos0;
   405	    s.end = run.pos0 + @as(i64, @intCast(run.n));
   406	}

exec
/bin/bash -lc "nl -ba src/dflash_load.zig | sed -n '190,252p'" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
   190	        lw.a_proj = try l.linear(&.{try layerName(&nb[1], i, "attention_conv.kernel_projection.weight")}, &.{KVW}, D, null);
   191	        lw.m_proj = try l.linear(&.{try layerName(&nb[1], i, "mlp_conv.kernel_projection.weight")}, &.{KVW}, D, null);
   192	    }
   193	    dr.fc = try l.linear(&.{"fc.weight"}, &.{D}, 5 * D, null);
   194	    dr.hproj = try l.linearB(&.{"candidate_selector.hidden_projection.weight"}, &.{df.RANK}, D, null);
   195	    dr.hnorm = try l.raw("hidden_norm.weight", D * 2);
   196	    dr.norm = try l.raw("norm.weight", D * 2);
   197	    if (l.at > total) return error.DrafterLayout;
   198	    var tail: Tail = .{ .loader = &l, .head_n = head.n };
   199	    const candidates = @import("dflash_candidates.zig").Layout.init(world, df.MAX_M, df.CW);
   200	    try @import("dflash_load_owner.zig").acquire(gpa, dr, &tail, Tail, .{ .book = df.VOCAB * df.RANK, .ctx = df.MAX_CTX, .cand = candidates.host, .hidden = df.MAX_M * df.RANK });
   201	    if (std.c.getenv("TF_GLM_DFLASH2_LOG")) |v| dr.log = std.fmt.parseInt(u64, std.mem.span(v), 10) catch 0;
   202	    if (std.c.getenv("TF_GLM_DFLASH2_PROF") != null) dr.prof = io;
   203	    std.log.info("dflash2: drafter from {s} (CC BY-NC-ND 4.0: non-commercial use only), linears at {d} bits, {d:.2} GiB on the device", .{ dir, if (bits == 0) 16 else bits, @as(f64, @floatFromInt(l.at)) / (1 << 30) });
   204	    return dr;
   205	}
   206	
   207	const Tail = struct {
   208	    loader: *Loader,
   209	    head_n: usize,
   210	
   211	    pub fn readBook(t: *Tail, i: usize, dst: []u16) !void {
   212	        const name = if (i == 0) "candidate_selector.predecessor_codebook" else "candidate_selector.successor_codebook";
   213	        @memcpy(std.mem.sliceAsBytes(dst), try t.loader.read(name, dst.len * 2));
   214	    }
   215	
   216	    pub fn scratch(t: *Tail, dr: *Drafter) !void {
   217	        try carve(dr, t.loader.w.d, t.head_n);
   218	    }
   219	
   220	    pub fn freeScratch(_: *Tail, dr: *Drafter) void {
   221	        dr.scratch.free();
   222	    }
   223	};
   224	
   225	/// The drafter's scratch, every field one carved piece of one buffer.
   226	fn carve(dr: *Drafter, d: *const cuda.Driver, head_n: usize) !void {
   227	    const M: u64 = df.MAX_M;
   228	    const C: u64 = df.MAX_CTX;
   229	    const chunks: u64 = (head_n + 4095) / 4096;
   230	    var part: u64 = 16;
   231	    for (dr.layers) |l| {
   232	        for ([_]df.Lin{ l.qkv, l.kv, l.o, l.gu, l.a_proj, l.m_proj }) |w| if (w == .bf16) {
   233	            part = @max(part, mm.partBytes(64, w.bf16));
   234	        };
   235	        part = @max(part, mm.partBytes(64, l.down));
   236	    }
   237	    if (dr.fc == .bf16) part = @max(part, mm.partBytes(64, dr.fc.bf16));
   238	    part = @max(part, mm.partBytes(64, dr.hproj));
   239	    const fields = .{
   240	        .{ "x", M * D * 2 },                                                                     .{ "y", M * D * 2 },                        .{ "y2", M * D * 2 },                .{ "dyn", M * KVW * 2 },
   241	        .{ "qkv", M * df.QKV * 2 },                                                              .{ "q", M * D * 2 },                        .{ "kb", M * KVW * 2 },              .{ "vb", M * KVW * 2 },
   242	        .{ "att", M * D * 2 },                                                                   .{ "a", M * D * 2 },                        .{ "gu", M * 2 * FF * 2 },           .{ "act", M * FF * 2 },
   243	        .{ "hn", M * D * 2 },                                                                    .{ "xs", M * (D / 64) * 4 },                .{ "g", M * df.RANK * 4 },           .{ "logits", M * head_n * 2 },
   244	        .{ "part", part },                                                                       .{ "gs", C * 5 * (D / 64) * 4 },            .{ "sums", 2 * M * D * 4 },          .{ "cin", C * df.TAP_ROW },
   245	        .{ "c1", C * D * 2 },                                                                    .{ "ctx", C * D * 2 },                      .{ "ckv", C * 2 * KVW * 2 },         .{ "taps_step", df.STEP_ROWS * df.TAP_ROW },
   246	        .{ "taps_prefill", C * df.TAP_ROW },                                                     .{ "ids", M * 4 },                          .{ "bpos", M * 4 },                  .{ "rings", df.MAX_S * 8 },
   247	        .{ "starts", df.MAX_S * 8 },                                                             .{ "froms", df.MAX_S * 8 },                 .{ "cpos", C * 8 },                  .{ "crings", C * 8 },
   248	        .{ "gtk", @import("dflash_candidates.zig").Layout.init(dr.world, M, df.CW).device * 4 }, .{ "pv", M * chunks * df.TOPK * 4 },        .{ "pi", M * chunks * df.TOPK * 4 }, .{ "pm", M * df.QH * df.CH * 4 },
   249	        .{ "pl", M * df.QH * df.CH * 4 },                                                        .{ "pacc", M * df.QH * df.CH * df.HD * 4 },
   250	    };
   251	    var at: u64 = 0;
   252	    inline for (fields) |f| {

exec
/bin/bash -lc "nl -ba src/cuda_taps.zig && nl -ba src/cuda_engine.zig | sed -n '435,485p'" in /home/zurih/codex-glm53-prefill/r1
 exited 1 in 0ms:
nl: src/cuda_taps.zig: No such file or directory

exec
/bin/bash -lc "rg -n 'df_ctx_kv|ctx_kv|cpos|crings|WIN|2048|cos|sin|rope' src/dflash* src/*dflash*" in /home/zurih/codex-glm53-prefill/r1
exec
/bin/bash -lc "rg -n 'pairBody|exl3\\.run|moe|expert' src/cuda_forward*.zig src/cuda_moe*.zig" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
src/dflash.zig:12:pub const WIN: i64 = 2048; // the context window (and a ring's slots)
src/dflash.zig:30:pub const MAX_CTX: usize = 2048; // context rows one ingest takes (a prompt chunk's)
src/dflash.zig:35:pub const ring_bytes: u64 = LAYERS * 2 * @as(u64, @intCast(WIN)) * KVW * 2;
src/dflash.zig:48:    ctx_kv: cuda.Function,
src/dflash.zig:150:    cpos: u64 = 0,
src/dflash.zig:151:    crings: u64 = 0,
src/dflash.zig:165:    h_cpos: []i64 = &.{},
src/dflash.zig:166:    h_crings: []u64 = &.{},
src/dflash.zig:178:        g.free(dr.h_cpos);
src/dflash.zig:179:        g.free(dr.h_crings);
src/dflash.zig:185:    /// TF_GLM_DFLASH2_PROF: the stream idle, the time since `since` added to its phase; the new mark.
src/dflash.zig:186:    fn mark(dr: *Drafter, e: anytype, since: ?struct { i96, usize }) !i96 {
src/dflash.zig:190:        if (since) |s| dr.phase_ms[s[1]] += @as(f64, @floatFromInt(now - s[0])) / 1e6;
src/dflash.zig:227:        return @max(@max(s.first, s.from), end - WIN);
src/dflash.zig:287:                dr.h_cpos[row + j] = r.pos0 + @as(i64, @intCast(j));
src/dflash.zig:288:                dr.h_crings[row + j] = r.ring;
src/dflash.zig:292:        try block.upload(dr, e, dr.cpos, std.mem.sliceAsBytes(dr.h_cpos[0..N]));
src/dflash.zig:293:        try block.upload(dr, e, dr.crings, std.mem.sliceAsBytes(dr.h_crings[0..N]));
src/dflash.zig:371:    const keep_from = len -| @as(usize, @intCast(WIN));
src/dflash.zig:385:    if (start < len -| @as(usize, @intCast(WIN)) or row0 > MAX_CTX or rows > MAX_CTX - row0) return error.BadPromptContext;
src/dflash_load.zig:43:            return error.DrafterTensorMissing;
src/dflash_load.zig:247:        .{ "starts", df.MAX_S * 8 },                                                             .{ "froms", df.MAX_S * 8 },                 .{ "cpos", C * 8 },                  .{ "crings", C * 8 },
src/dflash_load.zig:43:            return error.DrafterTensorMissing;
src/dflash_load.zig:247:        .{ "starts", df.MAX_S * 8 },                                                             .{ "froms", df.MAX_S * 8 },                 .{ "cpos", C * 8 },                  .{ "crings", C * 8 },
src/dflash.zig:12:pub const WIN: i64 = 2048; // the context window (and a ring's slots)
src/dflash.zig:30:pub const MAX_CTX: usize = 2048; // context rows one ingest takes (a prompt chunk's)
src/dflash.zig:35:pub const ring_bytes: u64 = LAYERS * 2 * @as(u64, @intCast(WIN)) * KVW * 2;
src/dflash.zig:48:    ctx_kv: cuda.Function,
src/dflash.zig:150:    cpos: u64 = 0,
src/dflash.zig:151:    crings: u64 = 0,
src/dflash.zig:165:    h_cpos: []i64 = &.{},
src/dflash.zig:166:    h_crings: []u64 = &.{},
src/dflash.zig:178:        g.free(dr.h_cpos);
src/dflash.zig:179:        g.free(dr.h_crings);
src/dflash.zig:185:    /// TF_GLM_DFLASH2_PROF: the stream idle, the time since `since` added to its phase; the new mark.
src/dflash.zig:186:    fn mark(dr: *Drafter, e: anytype, since: ?struct { i96, usize }) !i96 {
src/dflash.zig:190:        if (since) |s| dr.phase_ms[s[1]] += @as(f64, @floatFromInt(now - s[0])) / 1e6;
src/dflash.zig:227:        return @max(@max(s.first, s.from), end - WIN);
src/dflash.zig:287:                dr.h_cpos[row + j] = r.pos0 + @as(i64, @intCast(j));
src/dflash.zig:288:                dr.h_crings[row + j] = r.ring;
src/dflash.zig:292:        try block.upload(dr, e, dr.cpos, std.mem.sliceAsBytes(dr.h_cpos[0..N]));
src/dflash.zig:293:        try block.upload(dr, e, dr.crings, std.mem.sliceAsBytes(dr.h_crings[0..N]));
src/dflash.zig:371:    const keep_from = len -| @as(usize, @intCast(WIN));
src/dflash.zig:385:    if (start < len -| @as(usize, @intCast(WIN)) or row0 > MAX_CTX or rows > MAX_CTX - row0) return error.BadPromptContext;

 exited 2 in 0ms:
rg: src/cuda_moe*.zig: No such file or directory (os error 2)
src/cuda_forward.zig:13:const experts = @import("cuda_experts.zig");
src/cuda_forward.zig:17:const moe = @import("cuda_moe.zig");
src/cuda_forward.zig:41:    experts: experts.Kernels,
src/cuda_forward.zig:42:    exl3: exl3.Kernels, // an EXL3 checkpoint's routed experts
src/cuda_forward.zig:58:        for (&m.mods, [_][]const u8{ k.glm_kda, k.glm_ops, k.experts, k.affine4_pack, k.glm_quant, k.qmm_group, k.glm_glue, k.glm_mm }) |*x, image| {
src/cuda_forward.zig:67:        m.experts = try experts.Kernels.load(d, m.mods[2], @intCast(try ctx.attribute(.multiprocessor_count)));
src/cuda_forward.zig:68:        errdefer m.experts.unload();
src/cuda_forward.zig:105:        m.experts.unload();
src/cuda_forward.zig:211:    side: ?moe.Side = null, // a decode window's MoE shared expert on a second stream (TF_GLM_SHARED_SIDE, default on)
src/cuda_forward.zig:212:    moe_fuse: bool = false, // a decode window's EXL3 MoE glue fused (TF_GLM_MOE_FUSE, default on in the engine)
src/cuda_forward.zig:310:            .moe => |m| {
src/cuda_forward.zig:311:                try moe.body(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, rows, b.prefill);
src/cuda_forward.zig:312:                if (out) |o| try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, 0, rows, b.prefill, o);
src/cuda_forward.zig:321:    fn moeKernels(f: *const Forward) moe.Kernels {
src/cuda_forward.zig:322:        return .{ .set = f.set(), .experts = &f.m.experts, .exl3 = &f.m.exl3, .put_slot = f.m.put_slot, .prof = f.prof, .side = if (f.ranks[0].b.prefill) null else f.side, .fuse = f.moe_fuse };
src/cuda_forward.zig:340:                .moe => |m| try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, lo, hi, b.prefill, out),
src/cuda_forward.zig:425:    pub const pairBody = @import("cuda_forward_pair.zig").pairBody;

codex
The key distinction is already clear: a paired step runs two forward streams of work, but the code appears to invoke the routed experts separately for each chunk. I’m checking the layer schedule and tap-buffer bounds now, including whether raising only `MAX_CTX` is sufficient.
exec
/bin/bash -lc "nl -ba src/cuda_forward.zig | sed -n '275,355p;360,430p'" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
   275	                    s.saves.g = st.save_g + kv * 4;
   276	                    s.saves.b = st.save_b + li * window_rows * st.heads * 4;
   277	                }
   278	                const conv = st.conv + li * 3 * st.conv_c * 2;
   279	                try kda_block.bodyProfile(f.set(), &f.m.kda, f.stream, k, s, b.normed, c.hidden, rows, conv, st.recAt(cur, li), !b.prefill and f.kda_wide, b.prefill, f.prof);
   280	                if (out) |o| try kda_block.fill(f.set(), f.stream, k, s, 0, rows, o);
   281	                if (b.prefill) st.cur[li] = 1 - cur; // a prompt chunk keeps every row: the layer commits now
   282	            },
   283	            .mla => |a| {
   284	                try dsa_block.body(.{ .set = f.set(), .ops = f.m.ops(), .prof = f.prof, .split = f.indexSplit(r) }, f.stream, a, st.caches[di], b.dsa, b.normed, c.hidden, rows, st.pos_dev, st.pos, nch, b.part_rows);
   285	                if (out) |o| try dsa_block.fill(f.set(), f.stream, a, b.dsa, 0, rows, o);
   286	            },
   287	        }
   288	    }
   289	
   290	    /// A prompt chunk's indexer rows split across the ranks (index_split) over NCCL; in-process ranks pick every row.
   291	    fn indexSplit(f: *const Forward, r: Rank) ?sparse.Split {
   292	        if (!f.index_split or !r.b.prefill) return null;
   293	        return switch (f.exchange) {
   294	            .local => null,
   295	            .nccl => |x| .{ .lib = x.lib, .comm = x.comm, .rank = r.w.share.rank, .world = r.w.share.world },
   296	        };
   297	    }
   298	
   299	    /// forward._ffn: the dense MLP or MoE block on b.normed into this rank's slot.
   300	    pub fn ffn(f: *const Forward, r: Rank, l: weights.Layer, rows: usize, out: ?u64) !void {
   301	        const stamp = prof_mod.span(f.prof, f.stream, .ffn);
   302	        defer stamp.end();
   303	        const c = &r.w.config;
   304	        const b = r.b;
   305	        switch (l.ffn) {
   306	            .dense => |m| {
   307	                try mlp.body(f.mlpKernels(), f.stream, m, b.dense, b.normed, c.hidden, rows, b.prefill);
   308	                if (out) |o| try mlp.fill(f.mlpKernels(), f.stream, m, b.dense, 0, rows, b.prefill, o);
   309	            },
   310	            .moe => |m| {
   311	                try moe.body(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, rows, b.prefill);
   312	                if (out) |o| try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, 0, rows, b.prefill, o);
   313	            },
   314	        }
   315	    }
   316	
   317	    fn mlpKernels(f: *const Forward) mlp.Kernels {
   318	        return .{ .set = f.set(), .linear = .{ .lane = &f.m.lane, .nvfp4_prompt = &f.m.prompt }, .copy_rows = f.m.copy_rows };
   319	    }
   320	
   321	    fn moeKernels(f: *const Forward) moe.Kernels {
   322	        return .{ .set = f.set(), .experts = &f.m.experts, .exl3 = &f.m.exl3, .put_slot = f.m.put_slot, .prof = f.prof, .side = if (f.ranks[0].b.prefill) null else f.side, .fuse = f.moe_fuse };
   323	    }
   324	
   325	    pub const Site = enum { mixer, ffn };
   326	
   327	    /// forward.out_proj's fill of rows lo..hi at a site of layer `l` into `out` (this rank's rows of a piece).
   328	    pub fn fillRows(f: *const Forward, r: Rank, l: weights.Layer, site: Site, lo: usize, hi: usize, out: u64) !void {
   329	        const stamp = prof_mod.span(f.prof, f.stream, .output_projection);
   330	        defer stamp.end();
   331	        const c = &r.w.config;
   332	        const b = r.b;
   333	        switch (site) {
   334	            .mixer => switch (l.mixer) {
   335	                .kda => |k| try kda_block.fill(f.set(), f.stream, k, b.kda, lo, hi, out),
   336	                .mla => |a| try dsa_block.fill(f.set(), f.stream, a, b.dsa, lo, hi, out),
   337	            },
   338	            .ffn => switch (l.ffn) {
   339	                .dense => |m| try mlp.fill(f.mlpKernels(), f.stream, m, b.dense, lo, hi, b.prefill, out),
   340	                .moe => |m| try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, lo, hi, b.prefill, out),
   341	            },
   342	        }
   343	    }
   344	
   345	    /// Piece [lo, hi)'s room in a rank's gathered buffer (pieces never share bytes): [world][hi - lo][dims] fp32.
   346	    pub fn room(r: Rank, lo: usize) u64 {
   347	        return r.b.gathered + r.w.share.world * lo * r.w.config.hidden * 4;
   348	    }
   349	
   350	    /// reduce.shares: [lo, hi) of the rows of a piece of `n` each rank owns: about equal, the first n % world one more.
   351	    pub fn share(n: usize, world: usize, rank: usize) [2]usize {
   352	        return cuda.rank_scatter.share(n, world, rank);
   353	    }
   354	
   355	    /// Whether piece [lo, lo + n) splits (TF_GLM_PREFILL_REDUCE=split): at least one row for every rank.
   360	    pub fn copy(f: *const Forward, stream: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
   361	        try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.handle), "cuMemcpyDtoDAsync");
   362	    }
   363	
   364	    const exchangePieces = @import("cuda_forward_pieces.zig").exchangePieces;
   365	
   366	    /// hc_post of rows lo..hi from partials `g` [world][hi - lo][D] (a gathered room or a split one's own rows).
   367	    fn postRows(f: *const Forward, r: Rank, lo: usize, hi: usize, g: u64) !void {
   368	        const c = &r.w.config;
   369	        const b = r.b;
   370	        const S: usize = c.hc;
   371	        const D: usize = c.hidden;
   372	        const x = b.x + lo * S * D * 2;
   373	        try glue.hcPost(f.set(), f.stream, x, x, g, b.post + lo * S * 4, b.comb + lo * S * S * 4, hi - lo, D, S, r.w.share.world);
   374	    }
   375	
   376	    /// forward._pre of rows lo..hi.
   377	    fn preRows(f: *const Forward, r: Rank, hc: weights.Hc, norm: glue.Norm, lo: usize, hi: usize) !void {
   378	        const c = &r.w.config;
   379	        const b = r.b;
   380	        const S: usize = c.hc;
   381	        const D: usize = c.hidden;
   382	        try glue.hcPre(f.set(), f.stream, b.x + lo * S * D * 2, hc.fn_w, hc.base, hc.scale, norm, b.normed + lo * D * 2, b.xs + lo * (D / 64) * 4, b.post + lo * S * 4, b.comb + lo * S * S * 4, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, hi - lo, D, S);
   383	    }
   384	
   385	    /// One piece's glue after a site: hc_post, then the next layer's hc_pre or (after the last layer) the stream mean.
   386	    pub fn glueRows(f: *const Forward, r: Rank, j: usize, site: Site, lo: usize, hi: usize, g: u64) !void {
   387	        const stamp = prof_mod.span(f.prof, f.stream, .glue);
   388	        defer stamp.end();
   389	        const c = &r.w.config;
   390	        const layers = r.w.layers;
   391	        try f.postRows(r, lo, hi, g);
   392	        if (site == .mixer) return f.preRows(r, layers[j].ffn_hc, layers[j].post_norm, lo, hi);
   393	        if (j + 1 < layers.len) return f.preRows(r, layers[j + 1].attn_hc, layers[j + 1].in_norm, lo, hi);
   394	        try glue.streamMean(f.set(), f.stream, r.b.x + lo * c.hc * c.hidden * 2, r.b.hidden + lo * c.hidden * 2, hi - lo, c.hidden, c.hc);
   395	    }
   396	
   397	    const gluePieces = @import("cuda_forward_pieces.zig").gluePieces;
   398	    const tapPieces = @import("cuda_forward_pieces.zig").tapPieces;
   399	    const shareTaps = @import("cuda_forward_pieces.zig").shareTaps;
   400	
   401	    /// forward.pieced_layers: a prompt chunk's layers with exchanges in row pieces; same kernels and bits as unpieced.
   402	    fn pieced(f: *const Forward, p: *const Pieces, R: usize, cut: []const [2]usize) !void {
   403	        const layers = f.ranks[0].w.layers;
   404	        for (f.ranks) |r| try f.pre(r, r.w.layers[0].attn_hc, r.w.layers[0].in_norm, R);
   405	        var li: usize = 0;
   406	        var di: usize = 0;
   407	        for (layers, 0..) |_, j| {
   408	            prof_mod.context(f.prof, j, null, .mixer);
   409	            for (f.ranks) |r| if (r.segs.len > 0) try shared.mixer(f, r, r.w.layers[j], li, di, R, null) else try f.mixer(r, r.w.layers[j], li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), null);
   410	            try f.exchangePieces(p, j, .mixer, cut);
   411	            try f.gluePieces(p, j, .mixer, cut);
   412	            prof_mod.context(f.prof, j, null, .ffn);
   413	            for (f.ranks) |r| try f.ffn(r, r.w.layers[j], R, null);
   414	            try f.exchangePieces(p, j, .ffn, cut);
   415	            try f.gluePieces(p, j, .ffn, cut);
   416	            if (f.taps) |t| try f.tapPieces(t, j, cut);
   417	            switch (layers[j].mixer) {
   418	                .kda => li += 1,
   419	                .mla => di += 1,
   420	            }
   421	        }
   422	        if (f.taps) |t| try f.shareTaps(t, cut);
   423	    }
   424	
   425	    pub const pairBody = @import("cuda_forward_pair.zig").pairBody;
   426	
   427	    /// forward.forward of `ids`: logits in b.logits; a prompt chunk's KDA layers committed, a window's at `commit`.
   428	    pub fn run(f: *const Forward, ids: []const i32) !void {
   429	        const R = ids.len;
   430	        for (f.ranks) |r| {

exec
/bin/bash -lc 'nl -ba src/dflash_block.zig' in /home/zurih/codex-glm53-prefill/r1
 exited 1 in 0ms:
nl: src/dflash_block.zig: No such file or directory

exec
/bin/bash -lc "nl -ba src/cuda_engine.zig | sed -n '435,485p'" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
   435	    /// Where a prompt pass of `len` ids starts: 0, or `from`'s position with its kept state restored.
   436	    fn prefillStart(e: *Engine, len: usize, from: ?Resume) !usize {
   437	        if (!promptFits(len, e.cur.st.capacity)) return error.PromptTooLong;
   438	        const r = from orelse return 0;
   439	        if (r.at == 0 or r.at >= len or r.at % fw.prefill_rows != 0) return error.BadResume;
   440	        try e.restoreState(r.id); // DFlash2's context too: its kept rows, or from r.at when it kept none
   441	        return r.at;
   442	    }
   443	
   444	    /// One chunk of a `len`-id prompt pass: `part` from `start` plus the next id; first token at the end, else null.
   445	    pub fn prefillChunk(e: *Engine, part: []const u32, start: usize, len: usize, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !?u32 {
   446	        const split = e.index_split; // this step's alone
   447	        e.index_split = false;
   448	        if ((from != null or start == 0) and try e.prefillStart(len, from) != start) return error.BadChunk;
   449	        if (start >= len or start % fw.prefill_rows != 0) return error.BadChunk;
   450	        const rows = @min(len - start, fw.prefill_rows);
   451	        if (part.len > rows + 1) return prompt_pair.prefill(e, part, start, len, cancel, marks, split);
   452	        if (part.len != @min(len - start, rows + 1)) return error.BadChunk;
   453	        var one: [1]fw.Rank = undefined;
   454	        var f = e.view(&e.bp, &one);
   455	        f.index_split = split;
   456	        if (start + rows <= len -| @as(usize, @intCast(dflash.WIN))) f.taps = null; // none read (dflash.promptRun)
   457	        const as_i32: []const i32 = @ptrCast(part);
   458	        if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
   459	        f.prof = e.prompt_profs[if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3];
   460	        const labels = [_][]const u8{ "prompt chunk to 64 rows", "prompt chunk to 256 rows", "prompt chunk to 1024 rows", "prompt chunk to 2048 rows" };
   461	        if (f.prof) |p| try p.beginPrompt(e.stream, labels[if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3], .{ .rank = e.share.rank, .slot = e.slot, .start = start, .rows = rows }, e.c.run, e.pieces.count);
   462	        if (f.prof) |p| e.profiling_prompt = p;
   463	        defer if (f.prof != null) {
   464	            e.profiling_prompt = null;
   465	        };
   466	        defer if (f.prof) |p| p.end();
   467	        errdefer if (f.prof) |p| {
   468	            p.prompt.?.invalid = true;
   469	        };
   470	        var stamp = prof_mod.span(f.prof, e.stream, .gate);
   471	        e.gate(cancel) catch |err| {
   472	            if (err == error.Cancelled) try snapshots.interrupted(e, marks, @intCast(start));
   473	            return err;
   474	        };
   475	        stamp.end();
   476	        if (f.prof) |p| p.mark(e.stream, .gate);
   477	        try f.run(as_i32[0..rows]);
   478	        e.index_splits += @intFromBool(split);
   479	        stamp = prof_mod.span(f.prof, e.stream, .prompt_score);
   480	        if (e.prompt_logprobs) |p| if (part.len > 1) try prompt_lp.run(e, p, part[1..]);
   481	        stamp.end();
   482	        stamp = prof_mod.span(f.prof, e.stream, .drafter);
   483	        if (e.drafter) |dr| try dr.prompt(e, e.slot, start, rows, len);
   484	        if (e.drafter) |dr| if (start + rows == len) try dr.ringDigest(e, e.slot, len);
   485	        stamp.end();

exec
/bin/bash -lc 'rg --files | head -n 100' in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
log
src/dflash.zig
src/exl3_experts.zig
src/cuda_prompt_pair.zig
src/cuda_exl3.zig
src/cuda_engine.zig
src/cuda_relay.zig
src/dflash_load.zig
src/cuda_state.zig
src/cuda_forward.zig
brief.md
0094-glm-cuda-a-prompt-chunk-takes-4096-rows.patch

exec
/bin/bash -lc 'nl -ba src/dflash_load_owner.zig | head -n 180' in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
nl: src/dflash_load_owner.zig: No such file or directory

exec
/bin/bash -lc "rg -n 'exl3\\.run|df_ctx_kv|df_attn_part|df_tap|pairBody|fn body\\(' ." in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
./log:398:    26	pub const CH: usize = 16; // key chunks of the chunked attention (df_attn_part)
./log:917:   135	    try fw.Forward.pairBody(&fa, &fb, ra, rb);
./log:1283:/bin/bash -lc "rg -n 'df_ctx_kv|ctx_kv|cpos|crings|WIN|2048|cos|sin|rope' src/dflash* src/*dflash*" in /home/zurih/codex-glm53-prefill/r1
./log:1285:/bin/bash -lc "rg -n 'pairBody|exl3\\.run|moe|expert' src/cuda_forward*.zig src/cuda_moe*.zig" in /home/zurih/codex-glm53-prefill/r1
./log:1350:src/cuda_forward.zig:425:    pub const pairBody = @import("cuda_forward_pair.zig").pairBody;
./src/dflash.zig:26:pub const CH: usize = 16; // key chunks of the chunked attention (df_attn_part)
./src/cuda_prompt_pair.zig:135:    try fw.Forward.pairBody(&fa, &fb, ra, rb);
./src/cuda_forward.zig:425:    pub const pairBody = @import("cuda_forward_pair.zig").pairBody;

exec
/bin/bash -lc 'nl -ba src/cuda_forward_pair.zig' in /home/zurih/codex-glm53-prefill/r1
 exited 1 in 0ms:
nl: src/cuda_forward_pair.zig: No such file or directory

exec
/bin/bash -lc 'cat brief.md' in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
# Question: is a 4,096-row prompt chunk really gated on reworking the DFlash2 drafter?

You are reviewing TensorFold's GLM-5.3-Flash CUDA engine (Zig). `src/` holds the relevant files, copied from a
checkout of upstream `f350cd3e` with this recipe's engine patches 0001-0066 applied (patches 0067-0093 exist but are
not in these copies; none of them changes the constants below). Read-only: do not edit anything. Answer in `answer.md`.

## What I measured on two DGX Sparks (GB10, TP=2, one rank each)

`src/cuda_state.zig` has `pub const prefill_rows = 2048`. `src/cuda_exl3.zig` sets
`chunk_rows = cuda_state.prefill_rows` with a note that a call to the common EXL3 experts decodes **every** expert's
weights, so a prompt pass makes one call a chunk and that decode amortises over the chunk's rows; it records that at
1,024 rows "every trellis tile was decoded twice a pass" and names 4,096 as where upstream PR #1576 puts it.

I changed only that constant to 4,096 (`0094-...patch`) and measured, three cold trials a point, fresh random prompt
content each trial so no prefix cache is hit, `DRAFTER=none` on both sides so the drafter is out of the comparison:

| chunk rows | prefill @23.5K tok | @95K tok | NLL (4 passages) | sha of 10 greedy replies |
| --- | --- | --- | --- | --- |
| 2,048 | 1,626 tok/s | 1,648 tok/s | 0.7523 | 11ceddde19d4 |
| 4,096 | 1,743 tok/s (+7.2%) | 1,757 tok/s (+6.6%) | 0.7523 | 11ceddde19d4 |

So +7%, and bit-exact (identical NLL and identical greedy replies).

With the real config (`DRAFTER=dflash2`, the shipped default) the pair loads, passes a smoke test, and then the first
genuine prefill ends both ranks:

```
error: rank 0: rank 1 stopped (DrafterContextTooWide); this rank stops too
```

`src/dflash.zig` has `MAX_CTX = 2048` ("context rows one ingest takes (a prompt chunk's)") and `WIN = 2048`
("the context window (and a ring's slots)"), with the checks at lines ~282, ~370 and ~385.

## My current conclusion, which I want you to check

I concluded that getting the +7% needs a change to the drafter's ingest path (letting it take a wide chunk in
`MAX_CTX`-sized slices, since it only needs the last `WIN` rows), because `WIN` is the trained drafter's attention
window with rotary tables built for it, and so is not a buffer I can simply widen.

## Specific questions

1. **Is `MAX_CTX` alone the binding constraint?** If `MAX_CTX` were raised to 4,096 and `WIN` left at 2,048, would the
   result be correct, or does `WIN` genuinely bound how many rows of one chunk the drafter can ingest? Trace what
   `MAX_CTX` sizes (`dflash_load.zig` allocations, `gs`) versus what `WIN` governs (which taps/positions are read, the
   ring's slots, rotary). Cite line numbers.
2. **Does a paired prompt step already amortise the expert trellis decode over 4,096 rows?** `src/cuda_prompt_pair.zig`
   (TF_GLM_PROMPT_PAIR, on by default, `default_min_len = 2 * prefill_rows + 1`) makes a prompt step take two chunks.
   Does the EXL3 expert call then cover both chunks' rows in one call, or one call per chunk? If pairing already covers
   two chunks, why would raising `prefill_rows` still give +7%? This is the answer I most want.
3. **Is there a cheaper path to the same win** -- amortising the expert trellis decode over more rows *without*
   changing the chunk size the drafter ingests? For instance, making the experts call span a paired step's two chunks,
   or decoding the trellis once for two consecutive calls.
4. If a drafter-side change is genuinely required, what is the smallest correct one, and what would it risk (draft
   acceptance, `BadPromptContext` at ~385, the ring's slots)?

Be concrete and cite file:line. Say plainly if you think my conclusion is wrong.

exec
/bin/bash -lc "nl -ba src/exl3_experts.zig | sed -n '1,260p'" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
     1	//! The EXL3 routed experts (exl3/experts.cu's device code): the group, the grouped GEMMs, the epilogues.
     2	const std = @import("std");
     3	const driver = @import("driver.zig");
     4	const module = @import("module.zig");
     5	const launch_ = @import("launch.zig");
     6	const stream_ = @import("stream.zig");
     7	const kernels = @import("kernels.zig");
     8	const exl3 = @import("exl3.zig");
     9	const Driver = driver.Driver;
    10	const Module = module.Module;
    11	const Function = module.Function;
    12	const Stream = stream_.Stream;
    13	
    14	pub const DType = exl3.DType;
    15	pub const Codebook = exl3.Codebook;
    16	pub const prompt = @import("exl3_prompt.zig");
    17	
    18	/// One layer's routed experts (experts.py's Exl3RoutedExperts): per-expert trellis pointers, stacked scales, the clamp.
    19	pub const Layer = struct {
    20	    gate: u64, // int64 [E], one trellis pointer per expert
    21	    up: u64,
    22	    down: u64,
    23	    gate_k2: u64, // int32 [E], half-bits a value (2 * bits)
    24	    up_k2: u64,
    25	    down_k2: u64,
    26	    suh_g: u64, // fp16 [E, D]
    27	    suh_u: u64,
    28	    svh_g: u64, // fp16 [E, I]
    29	    svh_u: u64,
    30	    suh_d: u64, // fp16 [E, I]
    31	    svh_d: u64, // fp16 [E, D]
    32	    experts: usize, // E
    33	    dims: usize, // D (model width)
    34	    width: usize, // I (expert width)
    35	    codebook: Codebook,
    36	    uniform_k2: u32 = 0, // nonzero only when the loader verified this half-bit width for every projection
    37	    uniform_k2_gu: u32 = 0, // nonzero when the loader verified this half-bit width for every gate and up matrix
    38	    uniform_k2_d: u32 = 0, // nonzero when the loader verified this half-bit width for every down matrix
    39	    shared_input: bool = false, // the loader verified identical gate/up input scales across every expert
    40	    gu_stride: usize = 0, // gate/up trellis row stride in tiles (0: contiguous per projection)
    41	    limit: f32 = 0, // the SwiGLU clamp (0: none)
    42	    act_mode: i32 = 0, // 0: GLM's bf16 roundings (the route RED-SNOW runs), 1: fp32
    43	};
    44	
    45	/// What one projection launches (experts.py default_config): the 8-tile 4-warp K splits, shape-fixed; n % 128 required.
    46	pub fn plan(k: usize, n: usize, gateup: bool) !u32 {
    47	    const cands = [_]usize{ if (gateup) 4 else 1, 2, 1 };
    48	    for (cands) |sk| {
    49	        if (k % (16 * sk * 4) == 0 and n % 128 == 0) return @intCast(sk);
    50	    }
    51	    return error.TileNotBuilt;
    52	}
    53	
    54	/// One call's buffers (exl3/experts.py Scratch), laid out in scratch()'s order, each 256-byte aligned.
    55	pub const Scratch = struct {
    56	    xg: u64, // fp16 [P, D], the rotated gate inputs
    57	    xu: u64, // fp16 [P, D], the rotated up inputs
    58	    xd: u64, // fp16 [P, I], the rotated gate and up outputs (the down inputs)
    59	    z: u64, // fp32, gate and up write 2 * sk_gu * P * I partials, down sk_d * P * D
    60	    y: u64, // fp32 [P, D], the per-slot outputs
    61	    ids: u64, // int32 [min(P, E)], distinct routed experts in id order
    62	    count: u64, // int32 [1]
    63	    members: u64, // int32 [min(P, E), rows], row * 32 + slot, -1 after the last member
    64	    rows: usize, // the row capacity the buffers hold
    65	    slots: usize, // slots a row, at most 32
    66	};
    67	
    68	/// The byte size of the scratch buffers for a layer and row capacity, or error.TileNotBuilt when plan() fails.
    69	pub fn scratchBytes(l: Layer, rows: usize, slots: usize) !usize {
    70	    const g_sk = try plan(l.dims, l.width, true);
    71	    const d_sk = try plan(l.width, l.dims, false);
    72	    const p = rows * slots;
    73	    const maxu = @min(p, l.experts);
    74	    return align256(p * l.dims * 2) + align256(p * l.dims * 2) + align256(p * l.width * 2) +
    75	        align256(@max(2 * g_sk * l.width, d_sk * l.dims) * p * 4) + align256(p * l.dims * 4) +
    76	        align256(maxu * 4) + 256 + align256(maxu * rows * 4);
    77	}
    78	
    79	fn align256(n: usize) usize {
    80	    return (n + 255) / 256 * 256;
    81	}
    82	
    83	/// The pointers into one allocation of scratchBytes(), or error.TileNotBuilt when plan() fails.
    84	pub fn scratch(base: u64, l: Layer, rows: usize, slots: usize) !Scratch {
    85	    const p = rows * slots;
    86	    const g_sk = try plan(l.dims, l.width, true);
    87	    const d_sk = try plan(l.width, l.dims, false);
    88	    const maxu = @min(p, l.experts);
    89	    var at = base;
    90	    const take = struct {
    91	        fn f(a: *u64, bytes: usize) u64 {
    92	            const r = a.*;
    93	            a.* = r + align256(bytes);
    94	            return r;
    95	        }
    96	    }.f;
    97	    const xg = take(&at, p * l.dims * 2);
    98	    const xu = take(&at, p * l.dims * 2);
    99	    const xd = take(&at, p * l.width * 2);
   100	    const z = take(&at, @max(2 * g_sk * l.width, d_sk * l.dims) * p * 4);
   101	    const y = take(&at, p * l.dims * 4);
   102	    const ids = take(&at, maxu * 4);
   103	    const count = take(&at, 4);
   104	    const members = take(&at, maxu * rows * 4);
   105	    return .{ .xg = xg, .xu = xu, .xd = xd, .z = z, .y = y, .ids = ids, .count = count, .members = members, .rows = rows, .slots = slots };
   106	}
   107	
   108	/// The loaded kernels.
   109	pub const Experts = struct {
   110	    mod: Module,
   111	    group_fn: Function,
   112	    rot_in: Function,
   113	    grouped: [3]Function, // one per codebook
   114	    grouped_nt4: Function, // mcg at 4 tiles a block (the same bits as grouped[1])
   115	    grouped_r8: [2]Function, // mcg at one width (3-bit, 4-bit), 4 tiles, 8 reduction rows: windows of up to 8 rows
   116	    gateup: Function,
   117	    down_combine: Function,
   118	    prompt: prompt.Kernels = undefined,
   119	    plan_rotate: Function = undefined,
   120	    plan_finish: Function = undefined,
   121	
   122	    pub fn load(d: *const Driver) !Experts {
   123	        if (!kernels.available) return error.BuiltWithoutKernels;
   124	        var e: Experts = .{
   125	            .mod = try Module.load(d, kernels.exl3_experts),
   126	            .group_fn = undefined,
   127	            .rot_in = undefined,
   128	            .grouped = undefined,
   129	            .grouped_nt4 = undefined,
   130	            .grouped_r8 = undefined,
   131	            .gateup = undefined,
   132	            .down_combine = undefined,
   133	        };
   134	        errdefer e.mod.unload();
   135	        e.prompt = try prompt.Kernels.resolve(e.mod);
   136	        e.plan_rotate = try e.mod.function("tf_exl3_plan_rotate");
   137	        e.plan_finish = try e.mod.function("tf_exl3_plan_finish");
   138	        e.group_fn = try e.mod.function("tf_exl3_group");
   139	        try e.group_fn.allowDynamicShared(64 * 1024);
   140	        e.rot_in = try e.mod.function("tf_exl3_experts_rot_in");
   141	        e.grouped = .{
   142	            try e.mod.function("tf_exl3_grouped_0"),
   143	            try e.mod.function("tf_exl3_grouped_1"),
   144	            try e.mod.function("tf_exl3_grouped_2"),
   145	        };
   146	        e.grouped_nt4 = try e.mod.function("tf_exl3_grouped_1_nt4");
   147	        e.grouped_r8 = .{ try e.mod.function("tf_exl3_grouped_1_k6r8"), try e.mod.function("tf_exl3_grouped_1_k8r8") };
   148	        e.gateup = try e.mod.function("tf_exl3_gateup_epilogue");
   149	        e.down_combine = try e.mod.function("tf_exl3_down_combine");
   150	        return e;
   151	    }
   152	
   153	    pub fn unload(e: *Experts) void {
   154	        e.mod.unload();
   155	    }
   156	
   157	    /// Per-slot FP32 outputs after the same BF16-up contract; the caller owns its routing-weight reduction.
   158	    pub fn runSlots(e: *const Experts, s: Stream, l: Layer, x: u64, stride: usize, dtype: DType, pick: u64, rows: usize, sc: Scratch) !void {
   159	        try e.run(s, l, x, stride, dtype, pick, 0, rows, 0, sc);
   160	    }
   161	
   162	    /// The routed experts of `rows` rows: group, rotate in, grouped gate/up, SwiGLU, grouped down, wts-weighted sum.
   163	    pub fn run(e: *const Experts, s: Stream, l: Layer, x: u64, x_stride: usize, x_dtype: DType, pick: u64, wts: u64, rows: usize, out: u64, sc: Scratch) !void {
   164	        try e.group(s, l, pick, rows, sc);
   165	        try e.core(s, l, x, x_stride, x_dtype, pick, rows, sc);
   166	        try e.combine(s, l, pick, rows, sc, wts, sc.slots, 0, out);
   167	    }
   168	
   169	    /// `pick`'s distinct experts and their members into sc.ids, sc.count and sc.members (tf_exl3_group).
   170	    pub fn group(e: *const Experts, s: Stream, l: Layer, pick: u64, rows: usize, sc: Scratch) !void {
   171	        const g = try Geometry.of(l, rows, sc);
   172	        var a: launch_.Args = .{};
   173	        a.add(pick);
   174	        a.add(sc.ids);
   175	        a.add(sc.count);
   176	        a.add(sc.members);
   177	        a.add(i32d(rows));
   178	        a.add(i32d(g.slots));
   179	        a.add(i32d(l.experts));
   180	        a.add(i32d(rows));
   181	        try launch_.launch(e.group_fn, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 }, .shared = @intCast(rows * g.slots * 4) }, s, &a);
   182	    }
   183	
   184	    /// Rotate in, grouped gate/up, SwiGLU, grouped down: the down partials in sc.z from picks already grouped in sc
   185	    /// (`group`, or a caller's fused grouping with tf_exl3_group's own code, as GLM's decode router does).
   186	    pub fn core(e: *const Experts, s: Stream, l: Layer, x: u64, x_stride: usize, x_dtype: DType, pick: u64, rows: usize, sc: Scratch) !void {
   187	        const g = try Geometry.of(l, rows, sc);
   188	        var r: launch_.Args = .{};
   189	        r.add(x);
   190	        r.add(i32d(x_stride));
   191	        r.add(@as(i32, @backingInt(x_dtype)));
   192	        r.add(pick);
   193	        r.add(l.suh_g);
   194	        r.add(l.suh_u);
   195	        r.add(sc.xg);
   196	        r.add(sc.xu);
   197	        r.add(i32d(g.d));
   198	        r.add(i32d(g.slots));
   199	        r.add(i32d(l.experts));
   200	        try launch_.launch(e.rot_in, .{ .grid = .{ .x = @intCast(g.p), .y = @intCast(g.d / 128), .z = 2 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &r);
   201	
   202	        var gu: launch_.Args = .{};
   203	        gu.add(sc.xg);
   204	        gu.add(sc.xu);
   205	        gu.add(l.gate);
   206	        gu.add(l.up);
   207	        gu.add(l.gate_k2);
   208	        gu.add(l.up_k2);
   209	        gu.add(sc.ids);
   210	        gu.add(sc.count);
   211	        gu.add(sc.members);
   212	        gu.add(sc.z);
   213	        gu.add(i32d(g.d));
   214	        gu.add(i32d(g.w));
   215	        gu.add(i32d(g.p));
   216	        gu.add(i32d(g.g_sk));
   217	        gu.add(i32d(rows));
   218	        gu.add(i32d(g.slots));
   219	        gu.add(i32d(g.maxu));
   220	        gu.add(@as(i32, 2));
   221	        gu.add(i32d(l.gu_stride));
   222	        gu.add(@as(i32, 0));
   223	        gu.add(@as(i32, -1));
   224	        const gt = e.tilesFor(l, rows, .gate_up);
   225	        try launch_.launch(gt.f, .{ .grid = .{ .x = @intCast(g.maxu), .y = @intCast(g.w / (gt.nt * 16)), .z = @intCast(2 * g.g_sk * g.mt) }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &gu);
   226	
   227	        var ep: launch_.Args = .{};
   228	        ep.add(sc.z);
   229	        ep.add(pick);
   230	        ep.add(l.svh_g);
   231	        ep.add(l.svh_u);
   232	        ep.add(l.suh_d);
   233	        ep.add(sc.xd);
   234	        ep.add(i32d(g.p));
   235	        ep.add(i32d(g.w));
   236	        ep.add(i32d(g.g_sk));
   237	        ep.add(i32d(l.experts));
   238	        ep.add(l.limit);
   239	        ep.add(l.act_mode);
   240	        try launch_.launch(e.gateup, .{ .grid = .{ .x = @intCast(g.p), .y = @intCast(g.w / 128), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &ep);
   241	
   242	        var dn: launch_.Args = .{};
   243	        dn.add(sc.xd);
   244	        dn.add(sc.xd);
   245	        dn.add(l.down);
   246	        dn.add(l.down);
   247	        dn.add(l.down_k2);
   248	        dn.add(l.down_k2);
   249	        dn.add(sc.ids);
   250	        dn.add(sc.count);
   251	        dn.add(sc.members);
   252	        dn.add(sc.z);
   253	        dn.add(i32d(g.w));
   254	        dn.add(i32d(g.d));
   255	        dn.add(i32d(g.p));
   256	        dn.add(i32d(g.d_sk));
   257	        dn.add(i32d(rows));
   258	        dn.add(i32d(g.slots));
   259	        dn.add(i32d(g.maxu));
   260	        dn.add(@as(i32, 1));

exec
/bin/bash -lc "nl -ba src/cuda_state.zig | sed -n '1,90p'" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
     1	//! GLM-5.3-Flash's per-rank CUDA memory: forward Buffers (shared by sequences) and a sequence's State, an arena each.
     2	
     3	const std = @import("std");
     4	const cuda = @import("cuda");
     5	const exl3 = @import("cuda_exl3.zig");
     6	const weights = @import("cuda_weights.zig");
     7	const kda = @import("cuda_kda.zig");
     8	const b16 = @import("cuda_b16.zig");
     9	const latent = @import("cuda_latent.zig");
    10	const sparse = @import("cuda_sparse.zig");
    11	const experts = @import("cuda_experts.zig");
    12	const dsa_block = @import("cuda_dsa_block.zig");
    13	const moe = @import("cuda_moe.zig");
    14	const glue = @import("cuda_glue.zig");
    15	const mlp = @import("cuda_mlp.zig");
    16	const kda_block = @import("cuda_kda_block.zig");
    17	
    18	/// A prompt chunk's rows (decode.PREFILL_ROWS).
    19	pub const prefill_rows = 2048;
    20	/// A decode window's rows (Engine's max_rows).
    21	pub const window_rows = 16;
    22	/// Prompts one prompt pass takes together (cuda_prompts): a prompt buffer's logits rows, one a prompt.
    23	pub const prompt_heads = 16;
    24	/// The context the buffers' dense attention partials reach (Buffers' min(capacity, 2560)).
    25	pub const dense_reach = 2560;
    26	/// Index-key and gate rows a DSA cache holds (a ring): pool_keys reads only from the current pool's first row on.
    27	pub const index_ring = 2 * prefill_rows;
    28	
    29	/// Sub-allocations of one device allocation, 256-byte aligned like the driver's own.
    30	const Arena = struct {
    31	    buf: cuda.DeviceBuffer,
    32	    used: usize = 0,
    33	
    34	    fn take(a: *Arena, bytes: usize) u64 {
    35	        const at = std.mem.alignForward(usize, a.used, 256);
    36	        a.used = at + bytes;
    37	        std.debug.assert(a.used <= a.buf.len);
    38	        return a.buf.ptr + at;
    39	    }
    40	};
    41	
    42	/// Allocates `sizes` (bytes) in one zeroed device buffer and hands out each field's address in order.
    43	fn arena(d: *const cuda.Driver, sizes: []const usize, out: []u64) !Arena {
    44	    var total: usize = 0;
    45	    for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
    46	    var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, @max(total, 256)) };
    47	    errdefer a.buf.free();
    48	    try a.buf.fill8(0, null);
    49	    for (sizes, out) |n, *p| p.* = a.take(n);
    50	    return a;
    51	}
    52	
    53	/// Rows a KV page holds: one dense latent chunk (latent.chunk), and the keys of its 128 pools.
    54	pub const page_rows = latent.chunk;
    55	
    56	/// Where a paged sequence's caches are: the pool's pages at `kv`, `page_bytes` apart, through a table of `pages`.
    57	pub const Paged = struct { kv: u64, pages: usize, page_bytes: u64 };
    58	
    59	/// The DSA (MLA) layers among the loaded ones; the MTP layer's caches come after theirs.
    60	pub fn mlaLayers(w: *const weights.Weights) usize {
    61	    var nd: usize = 0;
    62	    for (w.layers) |l| nd += @intFromBool(l.mixer == .mla);
    63	    return nd;
    64	}
    65	
    66	/// One cache's part of a page: its latent rows (bf16, or E4M3 codes then their fp32 scales), then their pools' keys.
    67	pub fn pageCacheBytes(c: anytype, kv8: bool) usize {
    68	    const latent_row: usize = if (kv8) c.kv_lora + 4 else c.kv_lora * 2;
    69	    return page_rows * latent_row + page_rows / sparse.pool * c.i_dim * 2;
    70	}
    71	
    72	/// A page's bytes: every DSA layer's part, then the MTP layer's (13,056 a row on GLM-5.3-Flash).
    73	pub fn pageBytes(w: *const weights.Weights) u64 {
    74	    return pageBytesOf(w, false, true);
    75	}
    76	
    77	/// A page's bytes for this cache format: every DSA layer's part, then the target MTP head's if it is loaded
    78	/// (bf16 13,056 a row, E4M3 6,960; 11,968 and 6,380 without the head, as beside DFlash2).
    79	pub fn pageBytesOf(w: *const weights.Weights, kv8: bool, head: bool) u64 {
    80	    return (mlaLayers(w) + @intFromBool(head)) * pageCacheBytes(&w.config, kv8);
    81	}
    82	
    83	/// Cache `j`'s rows on pages: its part of page 0 at `p.kv`, the others through the table. In E4M3 its codes come
    84	/// first, then the part's scales (`ls`), then the pooled keys.
    85	fn onPages(x: *dsa_block.Caches, c: anytype, p: Paged, table: u64, j: usize, kv8: bool) void {
    86	    x.lc = p.kv + j * pageCacheBytes(c, kv8);
    87	    if (kv8) x.ls = x.lc + page_rows * c.kv_lora;
    88	    x.pk = if (kv8) x.ls + page_rows * 4 else x.lc + page_rows * c.kv_lora * 2;
    89	    x.pt = table;
    90	    x.ps = p.page_bytes;

exec
/bin/bash -lc 'cat 0094-glm-cuda-a-prompt-chunk-takes-4096-rows.patch' in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
From: MiaAI_Lab <mia.ailab1@gmail.com>
Subject: glm-cuda: a prompt chunk takes 4,096 rows, halving the trellis decode a row

A call to the common EXL3 experts decodes every expert's weights, and a prompt pass makes one call a chunk, so that
decode is a fixed cost amortised over the chunk's rows. cuda_exl3.zig's own note records the other direction of this
trade: at 1,024 rows "every trellis tile was decoded twice a pass", and it names 4,096 as where upstream's PR #1576
puts it. With 288 routed experts over 45 layers this is the largest fixed cost in a prompt pass on GLM-5.3-Flash.

Exact, not a numerics change: the same note states that a pair's sums do not depend on which item holds it, so the
bits are the same. The ship check is therefore identical greedy replies, not an NLL comparison.

What else follows prefill_rows: the prompt buffers and the int8 A8 prompt scratch are sized from it at load, and
index_ring is 2x it, so the chunk's device scratch roughly doubles; a chunk's rows also remain the step at which kept
prompt states are cut (cuda_lanes' SnapshotOutOfStep, prefillFrom's BadResume, prefillChunk's BadChunk), so a server
on this patch cannot resume a state cut by one without it. Both ranks must run the same value, which one image gives.

The prompt profiling buckets (cuda_engine's prompt_profs) are unchanged: their selector sends anything over 1,024
rows to the last bucket, so a 4,096-row chunk lands there safely, but that bucket's label still reads "to 2048 rows".
It is only visible under TF_GLM_PROF.

diff --git a/zig/src/families/glm/cuda_state.zig b/zig/src/families/glm/cuda_state.zig
--- a/zig/src/families/glm/cuda_state.zig
+++ b/zig/src/families/glm/cuda_state.zig
@@ -16,7 +16,7 @@
 const kda_block = @import("cuda_kda_block.zig");
 
 /// A prompt chunk's rows (decode.PREFILL_ROWS).
-pub const prefill_rows = 2048;
+pub const prefill_rows = 4096;
 /// A decode window's rows (Engine's max_rows).
 pub const window_rows = 16;
 /// Prompts one prompt pass takes together (cuda_prompts): a prompt buffer's logits rows, one a prompt.

exec
/bin/bash -lc "nl -ba src/dflash.zig | sed -n '295,365p'" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
   295	    }
   296	
   297	    /// Kept rows into each context and no block pass (a copy takes these streams' next windows): every rank.
   298	    pub fn ingestOnly(dr: *Drafter, e: anytype, reqs: []const Req) !void {
   299	        if (reqs.len > MAX_S) return error.TooManyStreams;
   300	        var runs: [MAX_S]Run = undefined;
   301	        try dr.ingest(e, runs[0..try dr.keptRuns(reqs, &runs)]);
   302	    }
   303	
   304	    /// Each request's kept verify rows as a context run (a context whose rows are gone restarts at its position).
   305	    fn keptRuns(dr: *Drafter, reqs: []const Req, runs: *[MAX_S]Run) !usize {
   306	        var nr: usize = 0;
   307	        for (reqs) |q| {
   308	            const s = try dr.slot(q.slot);
   309	            if (q.n == 0) continue;
   310	            if (s.v_step != dr.verifies or q.n > s.v_rows or s.v_pos0 + @as(i64, @intCast(q.n)) != q.pos) {
   311	                s.from = q.pos; // its verify rows were overwritten (or never written): its context restarts here
   312	                dr.lost += 1;
   313	                continue;
   314	            }
   315	            runs[nr] = .{ .ring = s.ring.?.ptr, .taps = dr.taps_step + s.v_row0 * TAP_ROW, .n = q.n, .pos0 = s.v_pos0 };
   316	            nr += 1;
   317	        }
   318	        return nr;
   319	    }
   320	
   321	    /// Kept rows into each context, then the block pass for every stream (every rank); rank 0 selects 7 drafts each.
   322	    pub fn draft(dr: *Drafter, e: anytype, reqs: []const Req) !void {
   323	        if (reqs.len == 0) return;
   324	        if (reqs.len > MAX_S) return error.TooManyStreams;
   325	        var tm = try dr.mark(e, null);
   326	        var runs: [MAX_S]Run = undefined;
   327	        try dr.ingest(e, runs[0..try dr.keptRuns(reqs, &runs)]);
   328	        tm = try dr.mark(e, .{ tm, 0 });
   329	        const S = reqs.len;
   330	        for (reqs, 0..) |q, i| {
   331	            const s = &dr.slots.items[q.slot];
   332	            dr.h_rings[i] = s.ring.?.ptr;
   333	            dr.h_starts[i] = q.pos;
   334	            dr.h_froms[i] = s.from;
   335	            for (0..BLOCK) |t| {
   336	                dr.h_ids[i * BLOCK + t] = @intCast(if (t == 0) q.anchor else MASK);
   337	                dr.h_bpos[i * BLOCK + t] = @intCast(q.pos + @as(i64, @intCast(t)));
   338	            }
   339	        }
   340	        const M = S * BLOCK;
   341	        try block.upload(dr, e, dr.ids, std.mem.sliceAsBytes(dr.h_ids[0..M]));
   342	        try block.upload(dr, e, dr.bpos, std.mem.sliceAsBytes(dr.h_bpos[0..M]));
   343	        try block.upload(dr, e, dr.rings, std.mem.sliceAsBytes(dr.h_rings[0..S]));
   344	        try block.upload(dr, e, dr.starts, std.mem.sliceAsBytes(dr.h_starts[0..S]));
   345	        try block.upload(dr, e, dr.froms, std.mem.sliceAsBytes(dr.h_froms[0..S]));
   346	        const rank0 = reqs[0].out != null;
   347	        tm = try dr.mark(e, .{ tm, 1 });
   348	        try block.pass(dr, e, S, rank0);
   349	        tm = try dr.mark(e, .{ tm, 2 });
   350	        if (rank0) {
   351	            try block.download(dr, e, std.mem.sliceAsBytes(dr.cand[0..@import("dflash_candidates.zig").Layout.init(dr.world, M, CW).read]), dr.gtk);
   352	            try block.download(dr, e, std.mem.sliceAsBytes(dr.gh[0 .. M * RANK]), dr.g);
   353	            for (reqs, 0..) |q, i| select.select(dr, i, M, q);
   354	            if (dr.log > dr.calls) for (reqs) |q| std.log.info("dflash2 round {d}: slot {d} pos {d} kept {d} anchor {d} drafts {any}", .{ dr.calls, q.slot, q.pos, q.n, q.anchor, q.out.?[0 .. BLOCK - 1] });
   355	            if (dr.calls % 256 == 255) std.log.info("dflash2: {d} rounds, {d} contexts restarted", .{ dr.calls + 1, dr.lost });
   356	        }
   357	        _ = try dr.mark(e, .{ tm, 3 });
   358	        dr.calls += 1;
   359	        if (dr.prof != null and dr.calls % 64 == 0) {
   360	            const k: f64 = 64;
   361	            std.log.info("dflash2 ms a round: ingest {d:.2} tables {d:.2} block {d:.2} select {d:.2}", .{ dr.phase_ms[0] / k, dr.phase_ms[1] / k, dr.phase_ms[2] / k, dr.phase_ms[3] / k });
   362	            dr.phase_ms = @splat(0);
   363	        }
   364	    }
   365	};

exec
/bin/bash -lc 'ls -la /home/zurih/codex-glm53-prefill /home/zurih/codex-glm53-prefill/r1' in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
/home/zurih/codex-glm53-prefill:
total 20
drwxrwxr-x   3 zurih zurih  4096 Nov  2 21:16 .
drwxr-x--- 116 zurih zurih 12288 Nov  2 21:16 ..
drwxrwxr-x   3 zurih zurih  4096 Nov  2 21:17 r1

/home/zurih/codex-glm53-prefill/r1:
total 144
drwxrwxr-x 3 zurih zurih   4096 Nov  2 21:17 .
drwxrwxr-x 3 zurih zurih   4096 Nov  2 21:16 ..
-rw-rw-r-- 1 zurih zurih   2077 Nov  2 21:16 0094-glm-cuda-a-prompt-chunk-takes-4096-rows.patch
-rw-rw-r-- 1 zurih zurih   3720 Nov  2 21:17 brief.md
-rw-rw-r-- 1 zurih zurih 126887 Nov  2 21:17 log
drwxrwxr-x 2 zurih zurih   4096 Nov  2 21:16 src

exec
/bin/bash -lc "rg -n 'pairBody|ctx_kv|rotary|rope|rotat|cuda_forward_pair|cuda_moe|attention window|MAX_CTX|decod.*trellis|ring' log | tail -n 100" in /home/zurih/codex-glm53-prefill/r1
 succeeded in 0ms:
1016:src/dflash.zig:202:        if (s.ring == null) s.ring = try cuda.DeviceBuffer.alloc(dr.d, ring_bytes);
1017:src/dflash.zig:253:        const host = try dr.gpa.alloc(u8, ring_bytes);
1024:src/dflash.zig:282:        if (N > MAX_CTX) return error.DrafterContextTooWide;
1029:src/dflash.zig:315:            runs[nr] = .{ .ring = s.ring.?.ptr, .taps = dr.taps_step + s.v_row0 * TAP_ROW, .n = q.n, .pos0 = s.v_pos0 };
1032:src/dflash.zig:370:    if (rows > MAX_CTX or row0 > MAX_CTX - rows) return error.DrafterContextTooWide;
1033:src/dflash.zig:374:    return .{ .ring = ring, .taps = taps + (row0 + lo) * TAP_ROW, .n = rows - lo, .pos0 = @intCast(start + lo) };
1050:   179	        g.free(dr.h_crings);
1073:   202	        if (s.ring == null) s.ring = try cuda.DeviceBuffer.alloc(dr.d, ring_bytes);
1086:   215	    /// Slot `i`'s ring (made on first use), for a kept state to copy out of or into.
1087:   216	    pub fn ringPtr(dr: *Drafter, i: u32) !u64 {
1088:   217	        return (try dr.slot(i)).ring.?.ptr;
1091:   220	    /// Where the context rows a state kept at `at` carries start: the prompt rows in the ring right up to `at`, at most
1092:   221	    /// a window of them (null: the pass tapped none just before `at`, so the ring holds nothing a resume could use).
1097:   226	        if (s.ring == null or s.first == no_rows or s.end != end or s.first >= end) return null;
1101:   230	    /// A kept state's ring copied into slot `i`: its sequence's rows `from` .. `at` are in it, as a fresh pass left them.
1114:   243	        s.* = .{ .ring = s.ring, .from = s.from, .first = s.first, .end = s.end, .v_step = dr.verifies, .v_row0 = row0, .v_rows = rows, .v_pos0 = @intCast(pos0) };
1117:   246	    /// TF_GLM_DFLASH2_RING_DIGEST=1 (a check, off by default): once a prompt is in, slot `i`'s whole context ring hashed
1119:   248	    pub fn ringDigest(dr: *Drafter, e: anytype, i: u32, len: usize) !void {
1124:   253	        const host = try dr.gpa.alloc(u8, ring_bytes);
1126:   255	        try s.ring.?.download(0, host);
1127:   256	        std.log.info("dflash2: ring digest slot {d} after a {d}-id prompt: {x:0>16}", .{ i, len, std.hash.Wyhash.hash(0, host) });
1130:   259	    /// A prompt chunk on slot `i` (rows start.. of a `len`-id prompt): its rows within the last window into the ring.
1148:   277	    /// The runs' rows through fc, the hidden norm and each layer's k, v into their rings.
1153:   282	        if (N > MAX_CTX) return error.DrafterContextTooWide;
1159:   288	                dr.h_crings[row + j] = r.ring;
1164:   293	        try block.upload(dr, e, dr.crings, std.mem.sliceAsBytes(dr.h_crings[0..N]));
1170:   368	pub fn promptRun(ring: u64, taps: u64, start: usize, rows: usize, len: usize, row0: usize) !?Run {
1172:   370	    if (rows > MAX_CTX or row0 > MAX_CTX - rows) return error.DrafterContextTooWide;
1176:   374	    return .{ .ring = ring, .taps = taps + (row0 + lo) * TAP_ROW, .n = rows - lo, .pos0 = @intCast(start + lo) };
1187:   385	    if (start < len -| @as(usize, @intCast(WIN)) or row0 > MAX_CTX or rows > MAX_CTX - row0) return error.BadPromptContext;
1194:   392	/// A prompt chunk's rows in the last window from tap rows at `taps` (row `row0` on) into slot `i`'s ring, and the run
1195:   393	/// the ring then holds (keptFrom's), whichever buffer they were tapped into.
1198:   396	    const run = (try promptRun(s.ring.?.ptr, taps, start, rows, len, row0)) orelse return;
1203:   401	/// The run slot `s`'s ring holds once `run` is in: it goes on from `end`, or starts again past a gap (rows a kept
1223:   200	    try @import("dflash_load_owner.zig").acquire(gpa, dr, &tail, Tail, .{ .book = df.VOCAB * df.RANK, .ctx = df.MAX_CTX, .cand = candidates.host, .hidden = df.MAX_M * df.RANK });
1251:   228	    const C: u64 = df.MAX_CTX;
1269:   246	        .{ "taps_prefill", C * df.TAP_ROW },                                                     .{ "ids", M * 4 },                          .{ "bpos", M * 4 },                  .{ "rings", df.MAX_S * 8 },
1270:   247	        .{ "starts", df.MAX_S * 8 },                                                             .{ "froms", df.MAX_S * 8 },                 .{ "cpos", C * 8 },                  .{ "crings", C * 8 },
1283:/bin/bash -lc "rg -n 'df_ctx_kv|ctx_kv|cpos|crings|WIN|2048|cos|sin|rope' src/dflash* src/*dflash*" in /home/zurih/codex-glm53-prefill/r1
1285:/bin/bash -lc "rg -n 'pairBody|exl3\\.run|moe|expert' src/cuda_forward*.zig src/cuda_moe*.zig" in /home/zurih/codex-glm53-prefill/r1
1287:src/dflash.zig:12:pub const WIN: i64 = 2048; // the context window (and a ring's slots)
1288:src/dflash.zig:30:pub const MAX_CTX: usize = 2048; // context rows one ingest takes (a prompt chunk's)
1289:src/dflash.zig:35:pub const ring_bytes: u64 = LAYERS * 2 * @as(u64, @intCast(WIN)) * KVW * 2;
1290:src/dflash.zig:48:    ctx_kv: cuda.Function,
1292:src/dflash.zig:151:    crings: u64 = 0,
1294:src/dflash.zig:166:    h_crings: []u64 = &.{},
1296:src/dflash.zig:179:        g.free(dr.h_crings);
1302:src/dflash.zig:288:                dr.h_crings[row + j] = r.ring;
1304:src/dflash.zig:293:        try block.upload(dr, e, dr.crings, std.mem.sliceAsBytes(dr.h_crings[0..N]));
1306:src/dflash.zig:385:    if (start < len -| @as(usize, @intCast(WIN)) or row0 > MAX_CTX or rows > MAX_CTX - row0) return error.BadPromptContext;
1308:src/dflash_load.zig:247:        .{ "starts", df.MAX_S * 8 },                                                             .{ "froms", df.MAX_S * 8 },                 .{ "cpos", C * 8 },                  .{ "crings", C * 8 },
1310:src/dflash_load.zig:247:        .{ "starts", df.MAX_S * 8 },                                                             .{ "froms", df.MAX_S * 8 },                 .{ "cpos", C * 8 },                  .{ "crings", C * 8 },
1311:src/dflash.zig:12:pub const WIN: i64 = 2048; // the context window (and a ring's slots)
1312:src/dflash.zig:30:pub const MAX_CTX: usize = 2048; // context rows one ingest takes (a prompt chunk's)
1313:src/dflash.zig:35:pub const ring_bytes: u64 = LAYERS * 2 * @as(u64, @intCast(WIN)) * KVW * 2;
1314:src/dflash.zig:48:    ctx_kv: cuda.Function,
1316:src/dflash.zig:151:    crings: u64 = 0,
1318:src/dflash.zig:166:    h_crings: []u64 = &.{},
1320:src/dflash.zig:179:        g.free(dr.h_crings);
1326:src/dflash.zig:288:                dr.h_crings[row + j] = r.ring;
1328:src/dflash.zig:293:        try block.upload(dr, e, dr.crings, std.mem.sliceAsBytes(dr.h_crings[0..N]));
1330:src/dflash.zig:385:    if (start < len -| @as(usize, @intCast(WIN)) or row0 > MAX_CTX or rows > MAX_CTX - row0) return error.BadPromptContext;
1333:rg: src/cuda_moe*.zig: No such file or directory (os error 2)
1335:src/cuda_forward.zig:17:const moe = @import("cuda_moe.zig");
1350:src/cuda_forward.zig:425:    pub const pairBody = @import("cuda_forward_pair.zig").pairBody;
1353:The key distinction is already clear: a paired step runs two forward streams of work, but the code appears to invoke the routed experts separately for each chunk. I’m checking the layer schedule and tap-buffer bounds now, including whether raising only `MAX_CTX` is sufficient.
1503:   425	    pub const pairBody = @import("cuda_forward_pair.zig").pairBody;
1567:   484	        if (e.drafter) |dr| if (start + rows == len) try dr.ringDigest(e, e.slot, len);
1592:/bin/bash -lc "rg -n 'exl3\\.run|df_ctx_kv|df_attn_part|df_tap|pairBody|fn body\\(' ." in /home/zurih/codex-glm53-prefill/r1
1595:./log:917:   135	    try fw.Forward.pairBody(&fa, &fb, ra, rb);
1596:./log:1283:/bin/bash -lc "rg -n 'df_ctx_kv|ctx_kv|cpos|crings|WIN|2048|cos|sin|rope' src/dflash* src/*dflash*" in /home/zurih/codex-glm53-prefill/r1
1597:./log:1285:/bin/bash -lc "rg -n 'pairBody|exl3\\.run|moe|expert' src/cuda_forward*.zig src/cuda_moe*.zig" in /home/zurih/codex-glm53-prefill/r1
1598:./log:1350:src/cuda_forward.zig:425:    pub const pairBody = @import("cuda_forward_pair.zig").pairBody;
1600:./src/cuda_prompt_pair.zig:135:    try fw.Forward.pairBody(&fa, &fb, ra, rb);
1601:./src/cuda_forward.zig:425:    pub const pairBody = @import("cuda_forward_pair.zig").pairBody;
1604:/bin/bash -lc 'nl -ba src/cuda_forward_pair.zig' in /home/zurih/codex-glm53-prefill/r1
1606:nl: src/cuda_forward_pair.zig: No such file or directory
1641:`src/dflash.zig` has `MAX_CTX = 2048` ("context rows one ingest takes (a prompt chunk's)") and `WIN = 2048`
1642:("the context window (and a ring's slots)"), with the checks at lines ~282, ~370 and ~385.
1647:`MAX_CTX`-sized slices, since it only needs the last `WIN` rows), because `WIN` is the trained drafter's attention
1648:window with rotary tables built for it, and so is not a buffer I can simply widen.
1652:1. **Is `MAX_CTX` alone the binding constraint?** If `MAX_CTX` were raised to 4,096 and `WIN` left at 2,048, would the
1654:   `MAX_CTX` sizes (`dflash_load.zig` allocations, `gs`) versus what `WIN` governs (which taps/positions are read, the
1655:   ring's slots, rotary). Cite line numbers.
1658:   Does the EXL3 expert call then cover both chunks' rows in one call, or one call per chunk? If pairing already covers
1662:   or decoding the trellis once for two consecutive calls.
1664:   acceptance, `BadPromptContext` at ~385, the ring's slots)?
1726:    56	    xg: u64, // fp16 [P, D], the rotated gate inputs
1727:    57	    xu: u64, // fp16 [P, D], the rotated up inputs
1728:    58	    xd: u64, // fp16 [P, I], the rotated gate and up outputs (the down inputs)
1789:   119	    plan_rotate: Function = undefined,
1806:   136	        e.plan_rotate = try e.mod.function("tf_exl3_plan_rotate");
1832:   162	    /// The routed experts of `rows` rows: group, rotate in, grouped gate/up, SwiGLU, grouped down, wts-weighted sum.
1947:    13	const moe = @import("cuda_moe.zig");
1960:    26	/// Index-key and gate rows a DSA cache holds (a ring): pool_keys reads only from the current pool's first row on.
1961:    27	pub const index_ring = 2 * prefill_rows;
2041:index_ring is 2x it, so the chunk's device scratch roughly doubles; a chunk's rows also remain the step at which kept
2085:   315	            runs[nr] = .{ .ring = s.ring.?.ptr, .taps = dr.taps_step + s.v_row0 * TAP_ROW, .n = q.n, .pos0 = s.v_pos0 };
2102:   332	            dr.h_rings[i] = s.ring.?.ptr;
2113:   343	        try block.upload(dr, e, dr.rings, std.mem.sliceAsBytes(dr.h_rings[0..S]));

web search: 
web search: github TensorFold GLM 5.3 Flash cuda_forward_pair.zig ...
web search: 
web search: https://raw.githubusercontent.com/ashhart/TensorFold/f350cd3e/zig/src/families/glm/cuda_forward_pair.zig
web search: 
web search: https://github.com/ashhart/TensorFold/blob/f350cd3e/zig/src/families/glm/cuda_forward_pair.zig
web search: 
web search: "TensorFold" "df_ctx_kv" ...
codex
**Your conclusion is too strong:** a 4,096-row target chunk does not require a sliced DFlash2 ingest. The smallest candidate fix is to raise `MAX_CTX` to 4,096 while leaving `WIN` at 2,048. In the supplied code, `MAX_CTX` is both an input-buffer capacity and a pre-clipping guard; `WIN` determines which prompt positions are retained. A wide target chunk can therefore supply at most the last 2,048 relevant rows to the drafter.

The expert side also explains why prompt pairing does not by itself establish 4,096-row amortization: the EXL3 adapter caps **each common expert call** at `chunk_rows`, currently 2,048. The missing pair implementation prevents me from tracing the exact number of adapter invocations, but even a 4,096-row invocation through this adapter is split into two common calls.

I could not create the requested `answer.md` because the workspace is read-only. The review is below.

1. **`MAX_CTX` is the immediate constraint; `WIN` need not increase.**

   [`promptRun`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:368) checks `rows > MAX_CTX` **before** checking whether the chunk is outside the final window or clipping its prefix. Thus a 4,096-row chunk can raise `DrafterContextTooWide` even when it contributes no rows, or only 2,048 rows, to the drafter. For a valid prompt run, the subsequent calculation keeps positions from `len - WIN` onward, so the returned run has at most `WIN` rows.

   `MAX_CTX` also sizes real storage. The loader uses it for host context capacity and the device allocations for `cin`, `taps_prefill`, `c1`, `ctx`, `ckv`, positions and ring pointers. It also sizes `gs`, the 4-bit linear’s input group sums. See [`dflash_load.zig:200`](/home/zurih/codex-glm53-prefill/r1/src/dflash_load.zig:200), [`dflash_load.zig:225`](/home/zurih/codex-glm53-prefill/r1/src/dflash_load.zig:225), and [`dflash.zig:137`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:137). Raising the constant consequently increases scratch memory per rank; it is not just relaxing a check.

   That storage matters before ingest: a forward pass can write a whole relevant chunk’s taps into `taps_prefill` before `dr.prompt` applies the guard ([`cuda_engine.zig:454`](/home/zurih/codex-glm53-prefill/r1/src/cuda_engine.zig:454)). Removing the guard alone, or slicing only after the forward pass, would leave a possible out-of-bounds tap write.

   `WIN`, by contrast, defines the final prompt window and the ring allocation ([`dflash.zig:12`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:12), [`dflash.zig:34`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:34), [`dflash.zig:220`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:220)). The drafter checkpoint’s [configuration](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/main/config.json) specifies a 2,048 sliding window and a much larger maximum position count. The supplied files do **not** include the rotary or ring-write kernels, so I cannot verify the claim that rotary tables themselves are built for only 2,048 positions, or independently verify their indexing. There is no need to change `WIN` for this proposed fix.

2. **Pairing does not establish one 4,096-row expert decode.**

   A paired step constructs separate forward views and buffers for chunks `a` and `b`, then passes both to `pairBody` ([`cuda_prompt_pair.zig:110`](/home/zurih/codex-glm53-prefill/r1/src/cuda_prompt_pair.zig:110), [`cuda_prompt_pair.zig:135`](/home/zurih/codex-glm53-prefill/r1/src/cuda_prompt_pair.zig:135)). Its stated purpose is to hide each chunk’s exchanges under the other’s work. The drafter then ingests the chunks separately ([`cuda_prompt_pair.zig:138`](/home/zurih/codex-glm53-prefill/r1/src/cuda_prompt_pair.zig:138)).

   The decisive visible limit is [`cuda_exl3.zig:281`](/home/zurih/codex-glm53-prefill/r1/src/cuda_exl3.zig:281): `run` loops in `chunk_rows` increments and invokes the common prompt or grouped expert implementation for each segment. Since `chunk_rows = prefill_rows` ([line 223](/home/zurih/codex-glm53-prefill/r1/src/cuda_exl3.zig:223)), 4,096 rows passed through this adapter with a 2,048 setting produce two common calls. Raising the setting allows one common call for 4,096 rows, consistent with the measured gain.

   The qualification is that `cuda_forward_pair.zig` and `cuda_moe.zig` are **not in the supplied copies**. I cannot prove from these files whether `pairBody` invokes the adapter once for each chunk, or whether there is a separate bypass. The separate views and the adapter’s cap support the former explanation, but the exact pair-to-expert call chain should be checked in those files or with a call counter. The +7% measurement is consistent with better expert amortization; it does not isolate that cause from all other effects of changing chunk size, including buffer sizes, scheduling and pair eligibility.

3. **A wider expert operation with 2,048-row drafter chunks is possible in principle, but is a larger engine change.**

   One design would retain the two 2,048-row target chunks and their separate drafter ingests, while gathering both chunks’ ready MoE inputs and routes into a 4,096-row expert operation, then scattering outputs back. It would need a larger independent EXL3 call cap and scratch arena, as well as changes to the pair’s scheduling, output layout and synchronization. It must preserve causal dependencies between the chunks and may affect the exchange overlap that pairing currently provides. The missing pair and MoE files prevent a reliable assessment of how invasive that is.

   Decoding weights once for two calls is another kernel/design project, not a switch exposed by the shown interface: the common calls receive trellis pointers and per-call scratch ([`exl3_experts.zig:18`](/home/zurih/codex-glm53-prefill/r1/src/exl3_experts.zig:18), [`exl3_experts.zig:54`](/home/zurih/codex-glm53-prefill/r1/src/exl3_experts.zig:54)). Materializing decoded weights would trade decode work for memory and bandwidth; retaining decoded tiles across both batches would require kernel and scheduling changes. Neither route is demonstrated to recover the measured 7%.

4. **Recommended smallest change and validation.**

   Try the existing 4,096-row target patch plus **only `MAX_CTX = 4096`**, with `WIN = 2048`. The visible prompt logic will still retain only the final window. In the paired path, `promptIn` requires the second chunk to lie wholly in that window and its offset plus rows to fit `MAX_CTX` ([`dflash.zig:381`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:381)). The branch selecting it uses `a_kept > 0`, meaning the second chunk starts inside the final window; the combined retained rows therefore fit within `WIN` for a valid pair ([`cuda_prompt_pair.zig:118`](/home/zurih/codex-glm53-prefill/r1/src/cuda_prompt_pair.zig:118)). The `BadPromptContext` guard should remain.

   This is a code-based candidate, not a demonstrated end-to-end result. Check device memory headroom and run with DFlash2 enabled on paired and unpaired prompts around 2,048 and 4,096 boundaries, including tails, longer prompts, and supported resume/snapshot cases. Compare valid retained ring contents or the available ring digest, draft acceptance and generated outputs against the 2,048-row baseline, then measure prefill and decode performance. The matching NLL and reply hashes are useful evidence for your sampled **no-drafter** trials, but do not prove bit-exactness for all prompts or validate drafter acceptance.
tokens used
57,241
**Your conclusion is too strong:** a 4,096-row target chunk does not require a sliced DFlash2 ingest. The smallest candidate fix is to raise `MAX_CTX` to 4,096 while leaving `WIN` at 2,048. In the supplied code, `MAX_CTX` is both an input-buffer capacity and a pre-clipping guard; `WIN` determines which prompt positions are retained. A wide target chunk can therefore supply at most the last 2,048 relevant rows to the drafter.

The expert side also explains why prompt pairing does not by itself establish 4,096-row amortization: the EXL3 adapter caps **each common expert call** at `chunk_rows`, currently 2,048. The missing pair implementation prevents me from tracing the exact number of adapter invocations, but even a 4,096-row invocation through this adapter is split into two common calls.

I could not create the requested `answer.md` because the workspace is read-only. The review is below.

1. **`MAX_CTX` is the immediate constraint; `WIN` need not increase.**

   [`promptRun`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:368) checks `rows > MAX_CTX` **before** checking whether the chunk is outside the final window or clipping its prefix. Thus a 4,096-row chunk can raise `DrafterContextTooWide` even when it contributes no rows, or only 2,048 rows, to the drafter. For a valid prompt run, the subsequent calculation keeps positions from `len - WIN` onward, so the returned run has at most `WIN` rows.

   `MAX_CTX` also sizes real storage. The loader uses it for host context capacity and the device allocations for `cin`, `taps_prefill`, `c1`, `ctx`, `ckv`, positions and ring pointers. It also sizes `gs`, the 4-bit linear’s input group sums. See [`dflash_load.zig:200`](/home/zurih/codex-glm53-prefill/r1/src/dflash_load.zig:200), [`dflash_load.zig:225`](/home/zurih/codex-glm53-prefill/r1/src/dflash_load.zig:225), and [`dflash.zig:137`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:137). Raising the constant consequently increases scratch memory per rank; it is not just relaxing a check.

   That storage matters before ingest: a forward pass can write a whole relevant chunk’s taps into `taps_prefill` before `dr.prompt` applies the guard ([`cuda_engine.zig:454`](/home/zurih/codex-glm53-prefill/r1/src/cuda_engine.zig:454)). Removing the guard alone, or slicing only after the forward pass, would leave a possible out-of-bounds tap write.

   `WIN`, by contrast, defines the final prompt window and the ring allocation ([`dflash.zig:12`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:12), [`dflash.zig:34`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:34), [`dflash.zig:220`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:220)). The drafter checkpoint’s [configuration](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/main/config.json) specifies a 2,048 sliding window and a much larger maximum position count. The supplied files do **not** include the rotary or ring-write kernels, so I cannot verify the claim that rotary tables themselves are built for only 2,048 positions, or independently verify their indexing. There is no need to change `WIN` for this proposed fix.

2. **Pairing does not establish one 4,096-row expert decode.**

   A paired step constructs separate forward views and buffers for chunks `a` and `b`, then passes both to `pairBody` ([`cuda_prompt_pair.zig:110`](/home/zurih/codex-glm53-prefill/r1/src/cuda_prompt_pair.zig:110), [`cuda_prompt_pair.zig:135`](/home/zurih/codex-glm53-prefill/r1/src/cuda_prompt_pair.zig:135)). Its stated purpose is to hide each chunk’s exchanges under the other’s work. The drafter then ingests the chunks separately ([`cuda_prompt_pair.zig:138`](/home/zurih/codex-glm53-prefill/r1/src/cuda_prompt_pair.zig:138)).

   The decisive visible limit is [`cuda_exl3.zig:281`](/home/zurih/codex-glm53-prefill/r1/src/cuda_exl3.zig:281): `run` loops in `chunk_rows` increments and invokes the common prompt or grouped expert implementation for each segment. Since `chunk_rows = prefill_rows` ([line 223](/home/zurih/codex-glm53-prefill/r1/src/cuda_exl3.zig:223)), 4,096 rows passed through this adapter with a 2,048 setting produce two common calls. Raising the setting allows one common call for 4,096 rows, consistent with the measured gain.

   The qualification is that `cuda_forward_pair.zig` and `cuda_moe.zig` are **not in the supplied copies**. I cannot prove from these files whether `pairBody` invokes the adapter once for each chunk, or whether there is a separate bypass. The separate views and the adapter’s cap support the former explanation, but the exact pair-to-expert call chain should be checked in those files or with a call counter. The +7% measurement is consistent with better expert amortization; it does not isolate that cause from all other effects of changing chunk size, including buffer sizes, scheduling and pair eligibility.

3. **A wider expert operation with 2,048-row drafter chunks is possible in principle, but is a larger engine change.**

   One design would retain the two 2,048-row target chunks and their separate drafter ingests, while gathering both chunks’ ready MoE inputs and routes into a 4,096-row expert operation, then scattering outputs back. It would need a larger independent EXL3 call cap and scratch arena, as well as changes to the pair’s scheduling, output layout and synchronization. It must preserve causal dependencies between the chunks and may affect the exchange overlap that pairing currently provides. The missing pair and MoE files prevent a reliable assessment of how invasive that is.

   Decoding weights once for two calls is another kernel/design project, not a switch exposed by the shown interface: the common calls receive trellis pointers and per-call scratch ([`exl3_experts.zig:18`](/home/zurih/codex-glm53-prefill/r1/src/exl3_experts.zig:18), [`exl3_experts.zig:54`](/home/zurih/codex-glm53-prefill/r1/src/exl3_experts.zig:54)). Materializing decoded weights would trade decode work for memory and bandwidth; retaining decoded tiles across both batches would require kernel and scheduling changes. Neither route is demonstrated to recover the measured 7%.

4. **Recommended smallest change and validation.**

   Try the existing 4,096-row target patch plus **only `MAX_CTX = 4096`**, with `WIN = 2048`. The visible prompt logic will still retain only the final window. In the paired path, `promptIn` requires the second chunk to lie wholly in that window and its offset plus rows to fit `MAX_CTX` ([`dflash.zig:381`](/home/zurih/codex-glm53-prefill/r1/src/dflash.zig:381)). The branch selecting it uses `a_kept > 0`, meaning the second chunk starts inside the final window; the combined retained rows therefore fit within `WIN` for a valid pair ([`cuda_prompt_pair.zig:118`](/home/zurih/codex-glm53-prefill/r1/src/cuda_prompt_pair.zig:118)). The `BadPromptContext` guard should remain.

   This is a code-based candidate, not a demonstrated end-to-end result. Check device memory headroom and run with DFlash2 enabled on paired and unpaired prompts around 2,048 and 4,096 boundaries, including tails, longer prompts, and supported resume/snapshot cases. Compare valid retained ring contents or the available ring digest, draft acceptance and generated outputs against the 2,048-row baseline, then measure prefill and decode performance. The matching NLL and reply hashes are useful evidence for your sampled **no-drafter** trials, but do not prove bit-exactness for all prompts or validate drafter acceptance.
