// KDA C1 microbenchmark: the shipped three-kernel chain (prep -> step -> out, and the _segs forms) against a chain
// whose prep is merged into step (pstep -> out), each also as programmatic dependent launches (PDL). Byte-compares
// every output of every candidate with the shipped chain's, then times a 34-layer forward's worth of chains so the
// recurrent state (2 MiB a layer) is never L2-resident. Build: nvcc -O3 --fmad=false -arch=sm_121 -o bench bench_kda_c1.cu
#include "glm_kda.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------------------------------------------------
// C1 candidates. Same expressions as prep_kernel_body / step_kernel_body / out_kernel_body, so the same bits.

__device__ __forceinline__ void c1_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void c1_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// Block (head, part), WARPS warps of 4 value rows each (as step_kernel_body): every block computes the head's prep
// (conv + SiLU, decay, beta, the q/k norms) for its rows straight into the step's smem tiles - prep's arithmetic per
// item, thread for thread the same expressions - and runs the delta rule on them. Block y == 0 writes the replay
// saves (k, decay, beta, and all of v); the q scratch is not written (nothing reads it).
template <int WARPS, int TR, bool PDL>
__device__ __forceinline__ void pstep_kernel_body(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off, const __nv_bfloat16* __restrict__ A,
        int a_stride, const __nv_bfloat16* __restrict__ cs, const float* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias, float lower,
        const float* __restrict__ state_in, int rows, float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save,
        float* __restrict__ g_save, float* __restrict__ b_save, __nv_bfloat16* __restrict__ y_out,
        float* __restrict__ state_out) {
    constexpr int NT = WARPS * 32, VR = WARPS * 4, ITEMS = 3 * DK + DK;  // a row's prep items: q | k | v, then decay
    const int C = 3 * H * DK;
    const int h = blockIdx.x, v0 = blockIdx.y * VR;
    const int lw = threadIdx.x >> 5, lane = threadIdx.x & 31, warp = blockIdx.y * WARPS + lw;
    const bool saver = blockIdx.y == 0;
    __shared__ __align__(16) float kt[TR][DK];
    __shared__ __align__(16) float qt[TR][DK];
    __shared__ __align__(16) float gt[TR][DK];
    __shared__ __align__(16) float vt[TR][VR];
    __shared__ float bt[TR];
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
    // the layer's state: written by a launch long before the one this grid may overlap - before the wait
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    if (PDL) {
        c1_wait();
        c1_trigger();
    }
    const float decay_rate = expf(a_log[h]);
    for (int r0 = 0; r0 < rows; r0 += TR) {
        const int n = rows - r0 < TR ? rows - r0 : TR;
        __syncthreads();
        for (int x = threadIdx.x; x < n * ITEMS; x += NT) {
            const int rr = x / ITEMS, t = x - rr * ITEMS, r = r0 + rr;
            const size_t base = ((size_t)r * H + h) * DK;
            if (t < 3 * DK) {
                const int c = (t / DK) * H * DK + h * DK + (t % DK);
                float acc = 0.0f;
#pragma unroll
                for (int tap = 0; tap < TAPS; ++tap) {
                    const int at = r + tap;
                    const float xv = at < TAPS - 1 ? __bfloat162float(cs[(size_t)at * C + c])
                                                   : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * p_stride + c]);
                    acc = acc + cw[(size_t)c * TAPS + tap] * xv;
                }
                const float act = bf(acc / (1.0f + expf(-acc)));
                if (t < DK) qt[rr][t] = act;
                else if (t < 2 * DK) kt[rr][t - DK] = act;
                else {
                    const int vc = t - 2 * DK;
                    if (vc >= v0 && vc < v0 + VR) vt[rr][vc - v0] = act;  // bf16-exact already: what step reads back
                    if (saver) v_save[base + vc] = __float2bfloat16_rn(act);
                }
            } else {
                const int i = t - 3 * DK;
                const float a = __bfloat162float(A[(size_t)r * a_stride + h * DK + i]) + dt_bias[h * DK + i];
                const float g = expf(lower * sigmoidf_(decay_rate * a));
                gt[rr][i] = g;
                if (saver) g_save[base + i] = g;
                if (i == 0) {
                    const float beta = bf(sigmoidf_(__bfloat162float(P[(size_t)r * p_stride + b_off + h])));
                    bt[rr] = beta;
                    if (saver) b_save[r * H + h] = beta;
                }
            }
        }
        __syncthreads();
        for (int w = lw; w < 2 * n; w += WARPS) {  // prep's q (even) and k (odd) norms: a warp a row, lane l owns 4l..4l+3
            const int rr = w >> 1;
            const bool is_k = (w & 1) != 0;
            float* x = is_k ? kt[rr] : qt[rr];
            float x4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { x4[i] = x[lane * 4 + i]; ss = ss + x4[i] * x4[i]; }
            ss = warp_sum(ss);
            const float inv = 1.0f / sqrtf(ss + 1e-6f);
            const size_t base = ((size_t)(r0 + rr) * H + h) * DK;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float y = x4[i] * inv;
                if (!is_k) y = y * (1.0f / sqrtf((float)DK));
                x[lane * 4 + i] = y;
                if (saver && is_k) k_save[base + lane * 4 + i] = y;
            }
        }
        __syncthreads();
        for (int rr = 0; rr < n; ++rr) {
            const float4 k4 = reinterpret_cast<const float4*>(kt[rr])[lane];
            const float4 q4 = reinterpret_cast<const float4*>(qt[rr])[lane];
            const float4 g4 = reinterpret_cast<const float4*>(gt[rr])[lane];
            const float4 v4 = reinterpret_cast<const float4*>(vt[rr])[lw];
            float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
            float gg[4] = {g4.x, g4.y, g4.z, g4.w}, vv[4] = {v4.x, v4.y, v4.z, v4.w};
            update4(s, kk, gg, vv, bt[rr], lane);
            float part[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float o = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
                part[j] = o;
            }
            const float o = warp_sum4(part, lane);
            if ((lane & 7) == 0)
                y_out[((size_t)(r0 + rr) * H + h) * DV + warp * 4 + (lane >> 3)] = __float2bfloat16_rn(o);
        }
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
    }
}

// out_kernel_body as a dependent launch: the gate and the norm weight (the projection's rows and a weight, finished
// before the chain's first kernel started) before the wait, the read-out after it.
__device__ __forceinline__ void out_p_kernel_body(
        int H, const __nv_bfloat16* __restrict__ y_in, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out) {
    const int r = blockIdx.x, h = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float rinv;
    const size_t base = ((size_t)r * H + h) * DV;
    const float gate = __bfloat162float(G[(size_t)r * g_stride + h * DV + t]);
    const float nw = __bfloat162float(norm_w[t]);
    c1_wait();  // no trigger: whatever follows the chain launches when out has finished
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float y = __bfloat162float(y_in[base + lane * 4 + i]); ss = ss + y * y; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = __bfloat162float(y_in[base + t]) * rinv;
    const float yw = nw * yn;
    out[(size_t)r * H * DV + h * DV + t] = __float2bfloat16_rn(yw * sigmoidf_(gate));
}

// step_kernel_body as a dependent launch (the reference chain with PDL, for the comparison only).
template <int WARPS, int TR>
__device__ __forceinline__ void step_p_kernel_body(
        int H, const float* __restrict__ state_in, const float* __restrict__ q_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        int rows, __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    constexpr int NT = WARPS * 32, VR = WARPS * 4;
    const int h = blockIdx.x, v0 = blockIdx.y * VR;
    const int lw = threadIdx.x >> 5, lane = threadIdx.x & 31, warp = blockIdx.y * WARPS + lw;
    __shared__ __align__(16) float kt[TR][DK];
    __shared__ __align__(16) float qt[TR][DK];
    __shared__ __align__(16) float gt[TR][DK];
    __shared__ __align__(16) float vt[TR][VR];
    __shared__ float bt[TR];
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    c1_wait();
    c1_trigger();
    for (int r0 = 0; r0 < rows; r0 += TR) {
        const int n = rows - r0 < TR ? rows - r0 : TR;
        __syncthreads();
        for (int x = threadIdx.x; x < n * DK; x += NT) {
            const int rr = x / DK, c = x % DK;
            const size_t base = ((size_t)(r0 + rr) * H + h) * DK + c;
            kt[rr][c] = k_save[base];
            qt[rr][c] = q_in[base];
            gt[rr][c] = g_save[base];
        }
        for (int x = threadIdx.x; x < n * VR; x += NT) {
            const int rr = x / VR, c = x % VR;
            vt[rr][c] = __bfloat162float(v_save[((size_t)(r0 + rr) * H + h) * DV + v0 + c]);
        }
        if (threadIdx.x < n) bt[threadIdx.x] = b_save[(r0 + threadIdx.x) * H + h];
        __syncthreads();
        for (int rr = 0; rr < n; ++rr) {
            const float4 k4 = reinterpret_cast<const float4*>(kt[rr])[lane];
            const float4 q4 = reinterpret_cast<const float4*>(qt[rr])[lane];
            const float4 g4 = reinterpret_cast<const float4*>(gt[rr])[lane];
            const float4 v4 = reinterpret_cast<const float4*>(vt[rr])[lw];
            float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
            float gg[4] = {g4.x, g4.y, g4.z, g4.w}, vv[4] = {v4.x, v4.y, v4.z, v4.w};
            update4(s, kk, gg, vv, bt[rr], lane);
            float part[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float o = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
                part[j] = o;
            }
            const float o = warp_sum4(part, lane);
            if ((lane & 7) == 0)
                y_out[((size_t)(r0 + rr) * H + h) * DV + warp * 4 + (lane >> 3)] = __float2bfloat16_rn(o);
        }
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
    }
}

#define PSTEP_ARGS                                                                                                   \
    int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off, const __nv_bfloat16* __restrict__ A,       \
    int a_stride, const __nv_bfloat16* __restrict__ cs, const float* __restrict__ cw,                              \
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, float lower,                               \
    const float* __restrict__ state_in, int rows, float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save,  \
    float* __restrict__ g_save, float* __restrict__ b_save, __nv_bfloat16* __restrict__ y_out,                      \
    float* __restrict__ state_out
#define PSTEP_PASS H, P, p_stride, b_off, A, a_stride, cs, cw, a_log, dt_bias, lower, state_in, rows, k_save, v_save, g_save, b_save, y_out, state_out

extern "C" __global__ void __launch_bounds__(256) tf_glm_kda_pstep(PSTEP_ARGS) { pstep_kernel_body<8, 16, false>(PSTEP_PASS); }
extern "C" __global__ void __launch_bounds__(128) tf_glm_kda_pstep4(PSTEP_ARGS) { pstep_kernel_body<4, 16, false>(PSTEP_PASS); }
extern "C" __global__ void __launch_bounds__(64) tf_glm_kda_pstep2(PSTEP_ARGS) { pstep_kernel_body<2, 16, false>(PSTEP_PASS); }
extern "C" __global__ void __launch_bounds__(256) tf_glm_kda_pstep_p(PSTEP_ARGS) { pstep_kernel_body<8, 16, true>(PSTEP_PASS); }
extern "C" __global__ void __launch_bounds__(128) tf_glm_kda_pstep4_p(PSTEP_ARGS) { pstep_kernel_body<4, 16, true>(PSTEP_PASS); }
extern "C" __global__ void __launch_bounds__(64) tf_glm_kda_pstep2_p(PSTEP_ARGS) { pstep_kernel_body<2, 16, true>(PSTEP_PASS); }

extern "C" __global__ void __launch_bounds__(128) tf_glm_kda_out_p(
        int H, const __nv_bfloat16* __restrict__ y_in, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out) {
    out_p_kernel_body(H, y_in, G, g_stride, norm_w, eps, out);
}

#define STEP_ARGS                                                                                                   \
    int H, const float* __restrict__ state_in, const float* __restrict__ q_in, const float* __restrict__ k_save,   \
    const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,  \
    int rows, __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out
#define STEP_PASS H, state_in, q_in, k_save, v_save, g_save, b_save, rows, y_out, state_out
extern "C" __global__ void __launch_bounds__(256) tf_glm_kda_step_p(STEP_ARGS) { step_p_kernel_body<8, 16>(STEP_PASS); }
extern "C" __global__ void __launch_bounds__(128) tf_glm_kda_step4_p(STEP_ARGS) { step_p_kernel_body<4, 16>(STEP_PASS); }
extern "C" __global__ void __launch_bounds__(64) tf_glm_kda_step2_p(STEP_ARGS) { step_p_kernel_body<2, 16>(STEP_PASS); }

// prep as a dependent launch: nothing of its own before the wait (its inputs are the projection's rows).
extern "C" __global__ void __launch_bounds__(512) tf_glm_kda_prep_p(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off, const __nv_bfloat16* __restrict__ A,
        int a_stride, const __nv_bfloat16* __restrict__ cs, const float* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias, float lower, float* __restrict__ q_out,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save) {
    c1_wait();
    c1_trigger();
    prep_kernel_body(H, P, p_stride, b_off, A, a_stride, cs, cw, a_log, dt_bias, lower, q_out, k_save, v_save, g_save, b_save);
}

// The shared window's forms: z picks the segment, as tf_glm_kda_step_segs (WARPS 8).
extern "C" __global__ void __launch_bounds__(256) tf_glm_kda_pstep_segs(
        int H, int p_stride, int b_off, int a_stride, const float* __restrict__ cw, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, float lower, const KdaSegs t) {
    const KdaSeg& g = t.s[blockIdx.z];
    pstep_kernel_body<8, 16, false>(H, g.P, p_stride, b_off, g.A, a_stride, g.cs, cw, a_log, dt_bias, lower, g.state_in,
                                    g.rows, g.k_save, g.v_save, g.g_save, g.b_save, g.y, g.state_out);
}
extern "C" __global__ void __launch_bounds__(256) tf_glm_kda_pstep_segs_p(
        int H, int p_stride, int b_off, int a_stride, const float* __restrict__ cw, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, float lower, const KdaSegs t) {
    const KdaSeg& g = t.s[blockIdx.z];
    pstep_kernel_body<8, 16, true>(H, g.P, p_stride, b_off, g.A, a_stride, g.cs, cw, a_log, dt_bias, lower, g.state_in,
                                   g.rows, g.k_save, g.v_save, g.g_save, g.b_save, g.y, g.state_out);
}
extern "C" __global__ void __launch_bounds__(128) tf_glm_kda_out_segs_p(
        int H, int g_stride, const __nv_bfloat16* __restrict__ norm_w, float eps, const KdaSegs t) {
    const KdaSeg& g = t.s[blockIdx.z];
    if (static_cast<int>(blockIdx.x) >= g.rows) {  // every CTA waits before it exits
        c1_wait();
        return;
    }
    out_p_kernel_body(H, g.y, g.G, g_stride, norm_w, eps, g.out);
}
extern "C" __global__ void __launch_bounds__(256) tf_glm_kda_step_segs_p(int H, const KdaSegs t) {
    const KdaSeg& g = t.s[blockIdx.z];
    step_p_kernel_body<8, 16>(H, g.state_in, g.q, g.k_save, g.v_save, g.g_save, g.b_save, g.rows, g.y, g.state_out);
}
extern "C" __global__ void __launch_bounds__(512) tf_glm_kda_prep_segs_p(
        int H, int p_stride, int b_off, int a_stride, const float* __restrict__ cw, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, float lower, const KdaSegs t) {
    c1_wait();
    c1_trigger();
    const KdaSeg& g = t.s[blockIdx.z];
    if (static_cast<int>(blockIdx.x) >= g.rows) return;
    prep_kernel_body(H, g.P, p_stride, b_off, g.A, a_stride, g.cs, cw, a_log, dt_bias, lower, g.q, g.k_save, g.v_save,
                     g.g_save, g.b_save);
}

// ---------------------------------------------------------------------------------------------------------------------
// Host harness.

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

template <typename... KArgs, typename... Args>
static void launchk(bool pdl, dim3 grid, dim3 block, cudaStream_t s, void (*k)(KArgs...), Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...));
}

static const int H = 32, LAYERS = 34, MAXROWS = 32;
static const int N = H * DK, C = 3 * N, WIDTH = 3 * N + 2 * DK + H, B_OFF = 3 * N + 2 * DK;
static const float LOWER = -2.0f, EPS = 1e-5f;

struct Layer {
    __nv_bfloat16 *P, *A, *G, *cs, *v, *y, *out;
    float *cw, *state_in, *state_out, *k, *g, *b, *q;
    float *seg_in[4], *seg_out[4];  // the shared window's other streams' states: distinct for every layer x segment
    __nv_bfloat16* seg_cs[4];
};

static unsigned long long rng = 0x9E3779B97F4A7C15ull;
static float uni(float lo, float hi) {
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    const float u = (float)((rng >> 11) & ((1ull << 24) - 1)) / (float)(1 << 24);
    return lo + (hi - lo) * u;
}
static __nv_bfloat16* dev_bf(size_t n, float lo, float hi) {
    std::vector<__nv_bfloat16> h(n);
    for (auto& x : h) x = __float2bfloat16(uni(lo, hi));
    __nv_bfloat16* d;
    CK(cudaMalloc(&d, n * sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(d, h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    return d;
}
static float* dev_f(size_t n, float lo, float hi) {
    std::vector<float> h(n);
    for (auto& x : h) x = uni(lo, hi);
    float* d;
    CK(cudaMalloc(&d, n * sizeof(float)));
    CK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}
template <typename T>
static T* dev_zero(size_t n) {
    T* d;
    CK(cudaMalloc(&d, n * sizeof(T)));
    CK(cudaMemset(d, 0, n * sizeof(T)));
    return d;
}

static int wp_of(int rows) { return rows == 1 ? 2 : rows <= 8 ? 4 : 8; }

enum Variant { REF3, REF3_PDL, REF3_PDL_SO, REF3_PDL_TRIG, FUSED2, FUSED2_PDL, CHAIN1, NVAR };
static const char* vname[NVAR] = {"ref3", "ref3+PDL(all 3)", "ref3+PDL(step,out)", "prep_p plain+PDL(step,out)", "fused2", "fused2+PDL", "chain1"};

static void run_rows(Variant v, const Layer& L, int rows, cudaStream_t s, float* a_log, float* dt_bias, __nv_bfloat16* norm_w) {
    const dim3 grid_rh(rows, H);
    const int wp = wp_of(rows);
    const dim3 grid_step(H, DV / 4 / wp), block_step(wp * 32);
    switch (v) {
    case REF3:
        launchk(false, grid_rh, dim3(512), s, tf_glm_kda_prep, H, L.P, WIDTH, B_OFF, L.A, N, L.cs, L.cw, a_log, dt_bias, LOWER, L.q, L.k, L.v, L.g, L.b);
        if (wp == 2) launchk(false, grid_step, block_step, s, tf_glm_kda_step2, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        else if (wp == 4) launchk(false, grid_step, block_step, s, tf_glm_kda_step4, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        else launchk(false, grid_step, block_step, s, tf_glm_kda_step, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        launchk(false, grid_rh, dim3(DV), s, tf_glm_kda_out, H, L.y, L.G, N, norm_w, EPS, L.out);
        break;
    case REF3_PDL:
        launchk(true, grid_rh, dim3(512), s, tf_glm_kda_prep_p, H, L.P, WIDTH, B_OFF, L.A, N, L.cs, L.cw, a_log, dt_bias, LOWER, L.q, L.k, L.v, L.g, L.b);
        if (wp == 2) launchk(true, grid_step, block_step, s, tf_glm_kda_step2_p, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        else if (wp == 4) launchk(true, grid_step, block_step, s, tf_glm_kda_step4_p, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        else launchk(true, grid_step, block_step, s, tf_glm_kda_step_p, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        launchk(true, grid_rh, dim3(DV), s, tf_glm_kda_out_p, H, L.y, L.G, N, norm_w, EPS, L.out);
        break;
    case REF3_PDL_SO:
        launchk(false, grid_rh, dim3(512), s, tf_glm_kda_prep, H, L.P, WIDTH, B_OFF, L.A, N, L.cs, L.cw, a_log, dt_bias, LOWER, L.q, L.k, L.v, L.g, L.b);
        if (wp == 2) launchk(true, grid_step, block_step, s, tf_glm_kda_step2_p, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        else if (wp == 4) launchk(true, grid_step, block_step, s, tf_glm_kda_step4_p, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        else launchk(true, grid_step, block_step, s, tf_glm_kda_step_p, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        launchk(true, grid_rh, dim3(DV), s, tf_glm_kda_out_p, H, L.y, L.G, N, norm_w, EPS, L.out);
        break;
    case REF3_PDL_TRIG:  // prep_p launched without the attribute (its wait a no-op, its trigger live): no squatting on the matmul before it
        launchk(false, grid_rh, dim3(512), s, tf_glm_kda_prep_p, H, L.P, WIDTH, B_OFF, L.A, N, L.cs, L.cw, a_log, dt_bias, LOWER, L.q, L.k, L.v, L.g, L.b);
        if (wp == 2) launchk(true, grid_step, block_step, s, tf_glm_kda_step2_p, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        else if (wp == 4) launchk(true, grid_step, block_step, s, tf_glm_kda_step4_p, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        else launchk(true, grid_step, block_step, s, tf_glm_kda_step_p, H, L.state_in, L.q, L.k, L.v, L.g, L.b, rows, L.y, L.state_out);
        launchk(true, grid_rh, dim3(DV), s, tf_glm_kda_out_p, H, L.y, L.G, N, norm_w, EPS, L.out);
        break;
    case FUSED2:
        if (wp == 2) launchk(false, grid_step, block_step, s, tf_glm_kda_pstep2, H, L.P, WIDTH, B_OFF, L.A, N, L.cs, L.cw, a_log, dt_bias, LOWER, L.state_in, rows, L.k, L.v, L.g, L.b, L.y, L.state_out);
        else if (wp == 4) launchk(false, grid_step, block_step, s, tf_glm_kda_pstep4, H, L.P, WIDTH, B_OFF, L.A, N, L.cs, L.cw, a_log, dt_bias, LOWER, L.state_in, rows, L.k, L.v, L.g, L.b, L.y, L.state_out);
        else launchk(false, grid_step, block_step, s, tf_glm_kda_pstep, H, L.P, WIDTH, B_OFF, L.A, N, L.cs, L.cw, a_log, dt_bias, LOWER, L.state_in, rows, L.k, L.v, L.g, L.b, L.y, L.state_out);
        launchk(false, grid_rh, dim3(DV), s, tf_glm_kda_out, H, L.y, L.G, N, norm_w, EPS, L.out);
        break;
    case FUSED2_PDL:
        if (wp == 2) launchk(true, grid_step, block_step, s, tf_glm_kda_pstep2_p, H, L.P, WIDTH, B_OFF, L.A, N, L.cs, L.cw, a_log, dt_bias, LOWER, L.state_in, rows, L.k, L.v, L.g, L.b, L.y, L.state_out);
        else if (wp == 4) launchk(true, grid_step, block_step, s, tf_glm_kda_pstep4_p, H, L.P, WIDTH, B_OFF, L.A, N, L.cs, L.cw, a_log, dt_bias, LOWER, L.state_in, rows, L.k, L.v, L.g, L.b, L.y, L.state_out);
        else launchk(true, grid_step, block_step, s, tf_glm_kda_pstep_p, H, L.P, WIDTH, B_OFF, L.A, N, L.cs, L.cw, a_log, dt_bias, LOWER, L.state_in, rows, L.k, L.v, L.g, L.b, L.y, L.state_out);
        launchk(true, grid_rh, dim3(DV), s, tf_glm_kda_out_p, H, L.y, L.G, N, norm_w, EPS, L.out);
        break;
    case CHAIN1:
        launchk(false, dim3(H), dim3(1024), s, tf_glm_kda_chain, H, L.P, WIDTH, B_OFF, L.A, N, L.G, N, L.cs, L.cw, L.state_in, a_log, dt_bias, norm_w, EPS, LOWER, rows, L.out, L.state_out, L.k, L.v, L.g, L.b);
        break;
    default: break;
    }
}

// Segments: the shared-window forms, `n` streams with rows[] each, their rows at consecutive offsets of one window.
static void run_segs(Variant v, const std::vector<Layer>& Ls, int li, const std::vector<int>& rows, cudaStream_t s, float* a_log, float* dt_bias, __nv_bfloat16* norm_w, bool lone_inputs = false) {
    const Layer& L = Ls[li];
    KdaSegs t = {};
    int most = 0, row0 = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const int r = rows[i];
        // each segment: its own state pair (the next layers' buffers stand in for the other streams' states), its
        // own conv window, its rows at row0 of the window's P/A/G/saves/scratch
        KdaSeg& g = t.s[i];
        g.P = L.P + (size_t)row0 * WIDTH;
        g.A = L.A + (size_t)row0 * N;
        g.G = L.G + (size_t)row0 * N;
        g.cs = lone_inputs && i == 0 ? L.cs : L.seg_cs[i];  // the cross-check below runs one segment on the lone chain's inputs
        g.state_in = lone_inputs && i == 0 ? L.state_in : L.seg_in[i];
        g.state_out = lone_inputs && i == 0 ? L.state_out : L.seg_out[i];
        g.q = L.q + (size_t)row0 * N;
        g.k_save = L.k + (size_t)row0 * N;
        g.v_save = L.v + (size_t)row0 * N;
        g.g_save = L.g + (size_t)row0 * N;
        g.b_save = L.b + (size_t)row0 * H;
        g.y = L.y + (size_t)row0 * N;
        g.out = L.out + (size_t)row0 * N;
        g.rows = r;
        most = most > r ? most : r;
        row0 += r;
    }
    const unsigned z = (unsigned)rows.size();
    const dim3 grid_rh(most, H, z), grid_step(H, DV / 4 / 8, z);
    switch (v) {
    case REF3:
        launchk(false, grid_rh, dim3(512), s, tf_glm_kda_prep_segs, H, WIDTH, B_OFF, N, L.cw, a_log, dt_bias, LOWER, t);
        launchk(false, grid_step, dim3(256), s, tf_glm_kda_step_segs, H, t);
        launchk(false, grid_rh, dim3(DV), s, tf_glm_kda_out_segs, H, N, norm_w, EPS, t);
        break;
    case REF3_PDL:
        launchk(true, grid_rh, dim3(512), s, tf_glm_kda_prep_segs_p, H, WIDTH, B_OFF, N, L.cw, a_log, dt_bias, LOWER, t);
        launchk(true, grid_step, dim3(256), s, tf_glm_kda_step_segs_p, H, t);
        launchk(true, grid_rh, dim3(DV), s, tf_glm_kda_out_segs_p, H, N, norm_w, EPS, t);
        break;
    case REF3_PDL_SO:
        launchk(false, grid_rh, dim3(512), s, tf_glm_kda_prep_segs, H, WIDTH, B_OFF, N, L.cw, a_log, dt_bias, LOWER, t);
        launchk(true, grid_step, dim3(256), s, tf_glm_kda_step_segs_p, H, t);
        launchk(true, grid_rh, dim3(DV), s, tf_glm_kda_out_segs_p, H, N, norm_w, EPS, t);
        break;
    case REF3_PDL_TRIG:
        launchk(false, grid_rh, dim3(512), s, tf_glm_kda_prep_segs_p, H, WIDTH, B_OFF, N, L.cw, a_log, dt_bias, LOWER, t);
        launchk(true, grid_step, dim3(256), s, tf_glm_kda_step_segs_p, H, t);
        launchk(true, grid_rh, dim3(DV), s, tf_glm_kda_out_segs_p, H, N, norm_w, EPS, t);
        break;
    case FUSED2:
        launchk(false, grid_step, dim3(256), s, tf_glm_kda_pstep_segs, H, WIDTH, B_OFF, N, L.cw, a_log, dt_bias, LOWER, t);
        launchk(false, grid_rh, dim3(DV), s, tf_glm_kda_out_segs, H, N, norm_w, EPS, t);
        break;
    case FUSED2_PDL:
        launchk(true, grid_step, dim3(256), s, tf_glm_kda_pstep_segs_p, H, WIDTH, B_OFF, N, L.cw, a_log, dt_bias, LOWER, t);
        launchk(true, grid_rh, dim3(DV), s, tf_glm_kda_out_segs_p, H, N, norm_w, EPS, t);
        break;
    default: break;
    }
}

struct Snap { std::vector<unsigned char> out, so, k, v, g, b, segso; };
static Snap snap(const Layer& L, int rows, int nseg = 0) {
    Snap s;
    s.segso.resize((size_t)nseg * H * DK * DV * 4);
    for (int i = 0; i < nseg; ++i) CK(cudaMemcpy(s.segso.data() + (size_t)i * H * DK * DV * 4, L.seg_out[i], (size_t)H * DK * DV * 4, cudaMemcpyDeviceToHost));
    s.out.resize((size_t)rows * N * 2); s.so.resize((size_t)H * DK * DV * 4);
    s.k.resize((size_t)rows * N * 4); s.v.resize((size_t)rows * N * 2); s.g.resize((size_t)rows * N * 4); s.b.resize((size_t)rows * H * 4);
    CK(cudaMemcpy(s.out.data(), L.out, s.out.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.so.data(), L.state_out, s.so.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.k.data(), L.k, s.k.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.v.data(), L.v, s.v.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.g.data(), L.g, s.g.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.b.data(), L.b, s.b.size(), cudaMemcpyDeviceToHost));
    return s;
}
static std::string diff(const Snap& a, const Snap& b) {
    std::string r;
    if (a.out != b.out) r += " out";
    if (a.so != b.so) r += " state_out";
    if (a.k != b.k) r += " k";
    if (a.v != b.v) r += " v";
    if (a.g != b.g) r += " g";
    if (a.b != b.b) r += " b";
    if (a.segso != b.segso) r += " seg_state_out";
    return r.empty() ? "same bits" : "DIFFERS:" + r;
}
static void poison(const Layer& L) {
    CK(cudaMemset(L.out, 0xFF, (size_t)MAXROWS * N * 2));
    CK(cudaMemset(L.state_out, 0xFF, (size_t)H * DK * DV * 4));
    CK(cudaMemset(L.k, 0xFF, (size_t)MAXROWS * N * 4));
    CK(cudaMemset(L.v, 0xFF, (size_t)MAXROWS * N * 2));
    CK(cudaMemset(L.g, 0xFF, (size_t)MAXROWS * N * 4));
    CK(cudaMemset(L.b, 0xFF, (size_t)MAXROWS * H * 4));
    CK(cudaMemset(L.q, 0xFF, (size_t)MAXROWS * N * 4));
    CK(cudaMemset(L.y, 0xFF, (size_t)MAXROWS * N * 2));
    for (int i = 0; i < 4; ++i) CK(cudaMemset(L.seg_out[i], 0xFF, (size_t)H * DK * DV * 4));
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    int dev = 0; cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("%s sm_%d%d, %d SMs, L2 %d MiB; H=%d layers=%d reps=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize >> 20, H, LAYERS, reps);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    float* a_log = dev_f(H, -3.0f, 0.0f);
    float* dt_bias = dev_f(H * DK, -1.0f, 1.0f);
    __nv_bfloat16* norm_w = dev_bf(DV, 0.5f, 1.5f);
    std::vector<Layer> Ls(LAYERS);
    for (auto& L : Ls) {
        L.P = dev_bf((size_t)MAXROWS * WIDTH, -2.0f, 2.0f);
        L.A = dev_bf((size_t)MAXROWS * N, -2.0f, 2.0f);
        L.G = dev_bf((size_t)MAXROWS * N, -3.0f, 3.0f);
        L.cs = dev_bf((size_t)(TAPS - 1) * C, -2.0f, 2.0f);
        L.cw = dev_f((size_t)C * TAPS, -0.5f, 0.5f);
        L.state_in = dev_f((size_t)H * DK * DV, -0.5f, 0.5f);
        L.state_out = dev_zero<float>((size_t)H * DK * DV);
        L.k = dev_zero<float>((size_t)MAXROWS * N);
        L.g = dev_zero<float>((size_t)MAXROWS * N);
        L.q = dev_zero<float>((size_t)MAXROWS * N);
        L.b = dev_zero<float>((size_t)MAXROWS * H);
        L.v = dev_zero<__nv_bfloat16>((size_t)MAXROWS * N);
        L.y = dev_zero<__nv_bfloat16>((size_t)MAXROWS * N);
        L.out = dev_zero<__nv_bfloat16>((size_t)MAXROWS * N);
        for (int i = 0; i < 4; ++i) {
            L.seg_in[i] = dev_f((size_t)H * DK * DV, -0.5f, 0.5f);
            L.seg_out[i] = dev_zero<float>((size_t)H * DK * DV);
            L.seg_cs[i] = dev_bf((size_t)(TAPS - 1) * C, -2.0f, 2.0f);
        }
    }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    printf("\n== lone window (chainRows): us per layer-chain, mean over %d x %d layers; bits against ref3 on every layer\n", reps, LAYERS);
    const int rows_list[] = {1, 2, 3, 4, 6, 8, 11, 14, 16, 24};
    for (int rows : rows_list) {
        printf("rows=%2d wp=%d:", rows, wp_of(rows));
        std::vector<Snap> ref(LAYERS);
        for (int v = 0; v < NVAR; ++v) {
            for (auto& L : Ls) poison(L);
            for (int li = 0; li < LAYERS; ++li) run_rows((Variant)v, Ls[li], rows, s, a_log, dt_bias, norm_w);
            CK(cudaStreamSynchronize(s));
            CK(cudaGetLastError());
            std::string bits = "same bits";
            for (int li = 0; li < LAYERS; ++li) {
                Snap sn = snap(Ls[li], rows);
                if (v == 0) ref[li] = sn;
                else { std::string d = diff(ref[li], sn); if (d != "same bits") bits = d; }
            }
            for (int w = 0; w < 3; ++w) for (int li = 0; li < LAYERS; ++li) run_rows((Variant)v, Ls[li], rows, s, a_log, dt_bias, norm_w);
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < reps; ++r) for (int li = 0; li < LAYERS; ++li) run_rows((Variant)v, Ls[li], rows, s, a_log, dt_bias, norm_w);
            CK(cudaEventRecord(e1, s));
            CK(cudaStreamSynchronize(s));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            printf("  | %s %.2f us [%s]", vname[v], ms * 1000.0f / (reps * LAYERS), v == 0 ? "ref" : bits.c_str());
        }
        printf("\n");
    }

    printf("\n== shared window (chainSegs), 4 streams: us per layer-chain; bits against ref3 (segs) on every layer\n");
    const std::vector<std::vector<int>> segs_list = {{2, 2, 2, 2}, {3, 2, 3, 2}, {3, 3, 3, 3}, {4, 3, 4, 3}, {4, 4, 4, 4}, {2, 2}};
    for (const auto& rows : segs_list) {
        int tot = 0; for (int r : rows) tot += r;
        printf("segs=%zu rows=%2d:", rows.size(), tot);
        std::vector<Snap> ref(LAYERS);
        for (int v = 0; v < CHAIN1; ++v) {
            for (auto& L : Ls) poison(L);
            for (int li = 0; li < LAYERS; ++li) run_segs((Variant)v, Ls, li, rows, s, a_log, dt_bias, norm_w);
            CK(cudaStreamSynchronize(s));
            CK(cudaGetLastError());
            std::string bits = "same bits";
            for (int li = 0; li < LAYERS; ++li) {
                Snap sn = snap(Ls[li], tot, (int)rows.size());
                if (v == 0) ref[li] = sn;
                else { std::string d = diff(ref[li], sn); if (d != "same bits") bits = d; }
            }
            for (int w = 0; w < 3; ++w) for (int li = 0; li < LAYERS; ++li) run_segs((Variant)v, Ls, li, rows, s, a_log, dt_bias, norm_w);
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < reps; ++r) for (int li = 0; li < LAYERS; ++li) run_segs((Variant)v, Ls, li, rows, s, a_log, dt_bias, norm_w);
            CK(cudaEventRecord(e1, s));
            CK(cudaStreamSynchronize(s));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            printf("  | %s %.2f us [%s]", vname[v], ms * 1000.0f / (reps * LAYERS), v == 0 ? "ref" : bits.c_str());
        }
        printf("\n");
    }
    // The segs reference against the lone chain: the same segment run alone must give the same bits (the shipped claim).
    {
        const int rows = 3;
        for (auto& L : Ls) poison(L);
        for (int li = 0; li < LAYERS; ++li) run_rows(REF3, Ls[li], rows, s, a_log, dt_bias, norm_w);
        CK(cudaStreamSynchronize(s));
        Snap lone = snap(Ls[0], rows);
        for (auto& L : Ls) poison(L);
        for (int li = 0; li < LAYERS; ++li) run_segs(REF3_PDL_TRIG, Ls, li, {rows}, s, a_log, dt_bias, norm_w, true);
        CK(cudaStreamSynchronize(s));
        Snap seg = snap(Ls[0], rows);
        printf("\nprep_p plain + PDL(step,out) segs (1 segment, %d rows) against the lone ref3 chain: %s\n", rows, diff(lone, seg).c_str());
    }
    return 0;
}
