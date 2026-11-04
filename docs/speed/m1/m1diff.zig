// M1: the two drafter weight quantizers, on the real DFlash2 safetensors, in f64.
// A: tf_glm_quantize4 (glm_quant.cu): 64-groups, scale/bias bf16-rounded BEFORE quantizing, rint, clamp.
// B: qwen4_exp bf16.py quantize4 (0002): 32-groups, f32 scale during quantize, roundEven, bf16 stores.
// Each side's kernel dequantizes with its stored bf16 pair: w = q * s_bf + b_bf (f32 math).
// Emulated dots are fp64 sequential of bf16 x times dequantized w: the trees' fp32 fma chains sit
// at 1e-6 relative; the weight representation is the term under test.
const std = @import("std");

var gpa_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
const gpa = gpa_impl.allocator();

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

const Tensor = struct { name: []const u8, shape: [2]usize, data: []u16 };

fn tobf(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    const rounded: u32 = (bits + 0x8000 + ((bits >> 16) & 1)) & 0xFFFF0000;
    return @intCast(rounded >> 16);
}

fn bf16Rne(x: f32) u16 {
    return tobf(x);
}

// ---- A's quantizer (glm_quant.cu tf_glm_quantize4), 64-groups ----
fn quantizeA(row: []const u16, k: usize, words: []u32, scales: []u16, biases: []u16) void {
    const groups = k / 64;
    var g: usize = 0;
    while (g < groups) : (g += 1) {
        const v = row[g * 64 ..][0..64];
        var lo = bf16ToF32(v[0]);
        var hi = lo;
        for (v) |b| {
            const x = bf16ToF32(b);
            lo = @min(lo, x);
            hi = @max(hi, x);
        }
        const sb: f32 = bf16ToF32(tobf(@max((hi - lo) * (1.0 / 15.0), 1e-8)));
        const bb: f32 = bf16ToF32(tobf(lo));
        var word: u32 = 0;
        for (v, 0..) |b, j| {
            const q = @min(@max(@round((bf16ToF32(b) - bb) / sb), 0.0), 15.0);
            word |= @as(u32, @intFromFloat(q)) << @intCast(4 * (j % 8));
            if (j % 8 == 7) {
                words[(g * 8 + j / 8)] = word;
                word = 0;
            }
        }
        scales[g] = bf16Rne(@max((hi - lo) * (1.0 / 15.0), 1e-8));
        biases[g] = bf16Rne(lo);
        _ = &word;
    }
}

// variant: 64-groups, f32 unrounded scale at quantize time (isolates grouping from A's rounding)
fn quantizeC(row: []const u16, k: usize, words: []u32, scales: []u16, biases: []u16) void {
    const groups = k / 64;
    var g: usize = 0;
    while (g < groups) : (g += 1) {
        const v = row[g * 64 ..][0..64];
        var mn: f32 = std.math.inf(f32);
        var mx = -std.math.inf(f32);
        for (v) |b| {
            const x = bf16ToF32(b);
            mn = @min(mn, x);
            mx = @max(mx, x);
        }
        const s = @max((mx - mn) * (1.0 / 15.0), 1e-8);
        var q: [64]u32 = undefined;
        for (v, 0..) |b, j| q[j] = @intFromFloat(std.math.clamp(roundEven((bf16ToF32(b) - mn) / s), 0, 15));
        var word: u32 = 0;
        for (0..8) |w| {
            word = 0;
            for (0..8) |p| word |= q[w * 8 + p] << @intCast(4 * p);
            words[g * 8 + w] = word;
        }
        scales[g] = bf16Rne(s);
        biases[g] = bf16Rne(mn);
    }
}

fn dequantC(words: []const u32, scales: []const u16, biases: []const u16, k: usize, out: []f64) void {
    for (0..k) |j| {
        const g = j / 64;
        out[j] = @as(f64, bf16ToF32(scales[g])) * @as(f64, @floatFromInt(nibble(words, j))) + bf16ToF32(biases[g]);
    }
}

// ---- B's quantizer (0002 quantize4), 32-groups, f32 scale at quantize time ----
fn roundEven(x: f32) f32 {
    if (@abs(x - @trunc(x)) == 0.5) return 2 * @round(x / 2);
    return @round(x);
}

fn quantizeB(row: []const u16, k: usize, words: []u32, scales: []u16, biases: []u16) void {
    const kg = k / 32;
    const inv15: f32 = 1.0 / 15.0;
    var g: usize = 0;
    while (g < kg) : (g += 1) {
        const v = row[g * 32 ..][0..32];
        var mn: f32 = std.math.inf(f32);
        var mx = -std.math.inf(f32);
        for (v) |b| {
            const x = bf16ToF32(b);
            mn = @min(mn, x);
            mx = @max(mx, x);
        }
        const s = @max((mx - mn) * inv15, 1e-8);
        var q: [32]u32 = undefined;
        for (v, 0..) |b, j| {
            q[j] = @intFromFloat(std.math.clamp(roundEven((bf16ToF32(b) - mn) / s), 0, 15));
        }
        var word: u32 = 0;
        for (0..4) |w| {
            word = 0;
            for (0..8) |p| word |= q[w * 8 + p] << @intCast(4 * p);
            words[g * 4 + w] = word;
        }
        scales[g] = bf16Rne(s);
        biases[g] = bf16Rne(mn);
    }
}

fn nibble(words: []const u32, idx: usize) u32 {
    return (words[idx / 8] >> @intCast(4 * (idx % 8))) & 0xF;
}

// A dequant, f64: q * f32(s_bf) + f32(b_bf), groups of 64.
fn dequantA(words: []const u32, scales: []const u16, biases: []const u16, k: usize, out: []f64) void {
    for (0..k) |j| {
        const g = j / 64;
        out[j] = @as(f64, bf16ToF32(scales[g])) * @as(f64, @floatFromInt(nibble(words, j))) + bf16ToF32(biases[g]);
    }
}

// B dequant, f32 kernel math on f64: same form, groups of 32.
fn dequantB(words: []const u32, scales: []const u16, biases: []const u16, k: usize, out: []f64) void {
    for (0..k) |j| {
        const g = j / 32;
        out[j] = @as(f64, bf16ToF32(scales[g])) * @as(f64, @floatFromInt(nibble(words, j))) + bf16ToF32(biases[g]);
    }
}

// exact: the bf16 value itself
fn dequantExact(row: []const u16, out: []f64) void {
    for (row, 0..) |b, j| out[j] = bf16ToF32(b);
}

fn rmsOf(d: []const f64) f64 {
    var s: f64 = 0;
    for (d) |x| s += x * x;
    return @sqrt(s / @as(f64, @floatFromInt(d.len)));
}

fn rmsDiff(a: []const f64, b: []const f64) f64 {
    var s: f64 = 0;
    for (a, b) |x, y| {
        const d = x - y;
        s += d * d;
    }
    return @sqrt(s / @as(f64, @floatFromInt(a.len)));
}

fn maxDiff(a: []const f64, b: []const f64) f64 {
    var m: f64 = 0;
    for (a, b) |x, y| m = @max(m, @abs(x - y));
    return m;
}

// ---- safetensors ----
const Entry = struct { name: []const u8, dtype: []const u8, shape: [2]usize, off: usize, nbytes: usize };

const St = struct { file: std.Io.File, data: []Entry, base: usize };

fn loadHeader(io: std.Io, path: []const u8, allocator: std.mem.Allocator) !St {
    var dir = std.Io.Dir.cwd();
    const file = try dir.openFile(io, path, .{});
    var buf: [8]u8 = undefined;
    _ = try file.readPositionalAll(io, &buf, 0);
    const hlen = std.mem.readInt(u64, &buf, .little);
    const hb = try allocator.alloc(u8, @intCast(hlen));
    _ = try file.readPositionalAll(io, hb, 8);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, hb, .{});
    var list: std.ArrayList(Entry) = .empty;
    var it = parsed.value.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const obj = kv.value_ptr.*.object;
        const shapeV = obj.get("shape").?.array;
        const data_off = obj.get("data_offsets").?.array;
        const dtype = obj.get("dtype").?.string;
        try list.append(allocator, .{
            .name = try allocator.dupe(u8, kv.key_ptr.*),
            .dtype = try allocator.dupe(u8, dtype),
                                .shape = blk2: {
                        var sh = [2]usize{ 0, 0 };
                        for (shapeV.items, 0..) |dim, di| {
                            if (di < 2) sh[di] = @intCast(dim.integer);
                        }
                        break :blk2 sh;
                    },
            .off = @intCast(data_off.items[0].integer),
            .nbytes = @intCast(data_off.items[1].integer - data_off.items[0].integer),
        });
    }
    return .{ .file = file, .data = try list.toOwnedSlice(allocator), .base = 8 + @as(usize, @intCast(hlen)) };
}

fn readRows(io: std.Io, st: *const St, e: Entry, row: usize, allocator: std.mem.Allocator) ![]u16 {
    const n = e.shape[1];
    const want = e.shape[0] * 2 * n;
    const off = st.base + e.off + row * 2 * n;
    const buf = try allocator.alloc(u16, n);
    const raw = try allocator.alloc(u8, n * 2);
    _ = try st.file.readPositionalAll(io, raw, off);
    _ = want;

    for (0..n) |i| buf[i] = @as(u16, raw[2 * i]) | (@as(u16, raw[2 * i + 1]) << 8);
    return buf;
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const path_arg: []const u8 = if (argv.len > 1) argv[1] else "model.safetensors";
    const io = init.io;
    const st = try loadHeader(io, path_arg, gpa);
    std.debug.print("tensors: {d}\n", .{st.data.len});

    // deterministic activation: bf16-rounded gaussian-ish (xorshift), 16 rows x 4096, unit-ish scale
    const seed: u64 = 0x9E3779B97F4A7C15;
    var seed_state = seed;
    const rng = struct {
        fn next(s: *u64) u64 {
            var x = s.*;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.* = x;
            return x;
        }
    }.next;
    _ = &seed_state;
    const rows: usize = 16;
    const kmax: usize = 4096;
    const X = try gpa.alloc(u16, rows * kmax);
    for (X) |*x| {
        const ua2 = @as(f64, @floatFromInt(rng(&seed_state) % (1 << 20))) / (1 << 20);
        const ub2 = @as(f64, @floatFromInt(rng(&seed_state) % (1 << 20))) / (1 << 20);
        const gg = @sqrt(-2 * std.math.log(f64, std.math.e, @max(ua2, 1e-12))) * @cos(6.283185307 * ub2) * 0.5;
        x.* = bf16Rne(@floatCast(gg));
    }

    const names = [_][]const u8{
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "fc.weight",
    };
    for (names) |want| {
        var found: ?Entry = null;
        for (st.data) |e| if (std.mem.eql(u8, e.name, want)) {
            found = e;
        };
        const e = found orelse {
            std.debug.print("{s}: MISSING\n", .{want});
            continue;
        };
        const n = e.shape[0];
        const k = e.shape[1];
        if (!std.mem.eql(u8, e.dtype, "BF16")) {
            std.debug.print("{s}: dtype {s} unexpected\n", .{ want, e.dtype });
            continue;
        }
        var worstA: f64 = 0;
        var worstB: f64 = 0;
        var worstAB: f64 = 0;
        var sumA: f64 = 0;
        var sumB: f64 = 0;
        var sumAB: f64 = 0;
        var sumC: f64 = 0;
        var cnt: usize = 0;
        // per-row quantize/dequant stats on a sample of rows (16), then the dot test on row-block 0
        var sample: usize = 0;
        while (sample < 16 and sample < n) : (sample += 1) {
            const row = try readRows(io, &st, e, sample, gpa);
            const wordsA = try gpa.alloc(u32, k / 8);
            const wordsB = try gpa.alloc(u32, k / 8);
            const scA = try gpa.alloc(u16, k / 64);
            const biA = try gpa.alloc(u16, k / 64);
            const scB = try gpa.alloc(u16, k / 32);
            const biB = try gpa.alloc(u16, k / 32);
            quantizeA(row, k, wordsA, scA, biA);
            quantizeB(row, k, wordsB, scB, biB);
            const wA = try gpa.alloc(f64, k);
            const wB = try gpa.alloc(f64, k);
            const wE = try gpa.alloc(f64, k);
            dequantA(wordsA, scA, biA, k, wA);
            dequantB(wordsB, scB, biB, k, wB);
            dequantExact(row, wE);
            const wordsC = try gpa.alloc(u32, k / 8);
            const scC = try gpa.alloc(u16, k / 64);
            const biC = try gpa.alloc(u16, k / 64);
            quantizeC(row, k, wordsC, scC, biC);
            const wC = try gpa.alloc(f64, k);
            dequantC(wordsC, scC, biC, k, wC);
            const rA = rmsDiff(wA, wE);
            const rB = rmsDiff(wB, wE);
            const rAB = rmsDiff(wA, wB);
            const scale = rmsOf(wE);
            sumC += rmsDiff(wC, wE) / scale;
            sumA += rA / scale;
            sumB += rB / scale;
            sumAB += rAB / scale;
            worstA = @max(worstA, maxDiff(wA, wE));
            worstB = @max(worstB, maxDiff(wB, wE));
            worstAB = @max(worstAB, maxDiff(wA, wB));
            cnt += 1;
            if (sample == 0) {
                // the dot: y_t = sum_j x_tj * w_j, fp64, per engine's weights; 16 rows
                const yA = try gpa.alloc(f64, rows);
                const yB = try gpa.alloc(f64, rows);
                const yE = try gpa.alloc(f64, rows);
                @memset(yA, 0);
                @memset(yB, 0);
                @memset(yE, 0);
                var t: usize = 0;
                while (t < rows) : (t += 1) {
                    var a: f64 = 0;
                    var b: f64 = 0;
                    var ee: f64 = 0;
                    for (0..k) |j| {
                        const xv = @as(f64, bf16ToF32(X[t * kmax + (j % kmax)]));
                        a += xv * wA[j];
                        b += xv * wB[j];
                        ee += xv * wE[j];
                    }
                    yA[t] = a;
                    yB[t] = b;
                    yE[t] = ee;
                }
                const oy = rmsOf(yE);
                std.debug.print("    dot@row0: rms(yA-yE)={d:.6} rms(yB-yE)={d:.6} rms(yA-yB)={d:.6} (y rms {d:.3})\n", .{ rmsDiff(yA, yE), rmsDiff(yB, yE), rmsDiff(yA, yB), oy });
            }
        }
        std.debug.print("{s} [{d}x{d}]: relrms A64={d:.5} B32={d:.5} C64f32={d:.5} AB={d:.5}\n", .{ want, n, k, sumA / @as(f64, @floatFromInt(cnt)), sumB / @as(f64, @floatFromInt(cnt)), sumC / @as(f64, @floatFromInt(cnt)), sumAB / @as(f64, @floatFromInt(cnt)) });
    }
}

fn readRowsT(io: std.Io, st: *const St, e: Entry, row: usize) ![]u16 {
    const n = e.shape[1];
    const buf = try gpa.alloc(u16, n);
    const off = st.base + e.off + row * 2 * n;
    const raw2 = try gpa.alloc(u8, n * 2);
    _ = try st.file.readPositionalAll(io, raw2, off);
    for (0..n) |i| buf[i] = @as(u16, raw2[2 * i]) | (@as(u16, raw2[2 * i + 1]) << 8);
    return buf;
}
