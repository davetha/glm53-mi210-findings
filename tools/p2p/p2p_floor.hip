// What is the IRREDUCIBLE cost of one 8 KB all-reduce between these two MI210s?
//
// Context this bounds: the model does 90 all-reduces per decode step, 8 KB each
// (hidden 4096 x bf16 x 1 token). NCCL spends 10.46 ms on that = 116 us per exchange,
// while 720 KB at the link's measured 26 GB/s is 28 us for the WHOLE step. So the cost
// is protocol, not bytes. A rewrite is only justified if the hardware floor is far
// below 116 us; this measures that floor.
//
// The previous version of this file HUNG and printed nothing. Two causes, both fixed:
//   * stdout was block-buffered, so a hang in a kernel discarded every prior printf.
//     -> line-buffered stdout, so partial progress always survives.
//   * the spin cap was 2e9 polls, sized for cached loads. These polls are uncached PCIe
//     reads at ~1 us each, so the "watchdog" was a ~2000 s no-op.
//     -> cap is now small, the kernel records how far it got, and the host reports a
//        bail as DATA rather than as silence.
//
// Test 0 comes first and CANNOT hang: does a peer write land at all? If it does not,
// every spin below would deadlock forever and nothing else is meaningful.
#include <hip/hip_runtime.h>
#include <cstdio>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error: %s (line %d)\n", hipGetErrorString(e), __LINE__); \
    fflush(stdout); return 1; } } while (0)

// gfx90a: L2 is not coherent with a peer's PCIe write, so every poll must bypass it.
// `glc slc` does that per access. (A system-scope acquire instead compiles to
// buffer_invl2 and discards the whole cache -- verified in isa_probe.hip.)
__device__ __forceinline__ unsigned int ld_uncached(const unsigned int* p)
{
    unsigned int v;
    asm volatile("global_load_dword %0, %1, off glc slc\n\ts_waitcnt vmcnt(0)"
                 : "=v"(v) : "v"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_uncached(unsigned int* p, unsigned int v)
{
    asm volatile("global_store_dword %0, %1, off glc slc\n\ts_waitcnt vmcnt(0)"
                 : : "v"(p), "v"(v) : "memory");
}

// Poll through L2 (glc, no slc). If an incoming PCIe write is absorbed into the
// TARGET's L2, then an slc read bypasses L2 and sees stale HBM forever -- which is what
// the bail-at-iter-1 result suggests, and the opposite of the usual assumption.
__device__ __forceinline__ unsigned int ld_glc(const unsigned int* p)
{
    unsigned int v;
    asm volatile("global_load_dword %0, %1, off glc\n\ts_waitcnt vmcnt(0)"
                 : "=v"(v) : "v"(p) : "memory");
    return v;
}
// The textbook-correct read: system-scope acquire (compiles to glc + buffer_invl2).
__device__ __forceinline__ unsigned int ld_acq(const unsigned int* p)
{
    return __scoped_atomic_load_n(p, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_rel(unsigned int* p, unsigned int v)
{
    __scoped_atomic_store_n(p, v, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
}

// Bound the spin by TIME, not poll count. A count-based cap is unusable here: the poll
// rate is unknown (an uncached PCIe read could be 100 ns or 2 us), and the first
// iteration must also outlast the peer kernel's code-object load, which is the bug that
// made every arm bail at iter 1. s_memrealtime is a fixed-frequency counter, so a tick
// budget is a real wall-clock bound either way.
#define SPIN_TICKS 100000000ull   // ~1 s at 100 MHz, ~4 s at 25 MHz

// 1 on success, 0 if the flag never arrived within the time budget.
__device__ __forceinline__ int spin_until(const unsigned int* p, unsigned int want)
{
    unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
    while (ld_uncached(p) < want)
        if (__builtin_amdgcn_s_memrealtime() - t0 > SPIN_TICKS) return 0;
    return 1;
}

template <int LD, int ST>
__device__ __forceinline__ unsigned int ld_sel(const unsigned int* p)
{ return LD == 0 ? ld_uncached(p) : (LD == 1 ? ld_glc(p) : ld_acq(p)); }
template <int ST>
__device__ __forceinline__ void st_sel(unsigned int* p, unsigned int v)
{ if (ST == 0) st_uncached(p, v); else st_rel(p, v); }

// Which (store, load) pair actually lets one GPU observe the other's write while both
// kernels are RUNNING? This is the foundational fact for any custom all-reduce here.
template <int LD, int ST>
__global__ void probe_protocol(unsigned int* peer_inbox, unsigned int* my_inbox,
                               int iters, int is_initiator, unsigned int* done)
{
    if (threadIdx.x || blockIdx.x) return;
    for (int i = 1; i <= iters; ++i) {
        unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
        if (is_initiator) {
            st_sel<ST>(peer_inbox, i);
            while (ld_sel<LD, ST>(my_inbox) < (unsigned)i)
                if (__builtin_amdgcn_s_memrealtime() - t0 > SPIN_TICKS) { *done = i; return; }
        } else {
            while (ld_sel<LD, ST>(my_inbox) < (unsigned)i)
                if (__builtin_amdgcn_s_memrealtime() - t0 > SPIN_TICKS) { *done = i; return; }
            st_sel<ST>(peer_inbox, i);
        }
    }
    *done = 0;
}

// ---- Test 0: does a peer write land? Straight-line, no polling, cannot hang. ------
__global__ void poke_peer(unsigned int* peer_buf, unsigned int val)
{
    if (!threadIdx.x) st_uncached(peer_buf, val);
}

// ---- flag-only round trip --------------------------------------------------------
__global__ void pingpong_flag(unsigned int* peer_inbox, unsigned int* my_inbox,
                              int iters, int is_initiator, unsigned int* done)
{
    if (threadIdx.x || blockIdx.x) return;
    for (int i = 1; i <= iters; ++i) {
        if (is_initiator) {
            st_uncached(peer_inbox, i);
            if (!spin_until(my_inbox, i)) { *done = i; return; }
        } else {
            if (!spin_until(my_inbox, i)) { *done = i; return; }
            st_uncached(peer_inbox, i);
        }
    }
    *done = 0;   // 0 == completed every iteration
}

// ---- PUSH: write payload into the peer, then flag, then wait ---------------------
__global__ void pingpong_push(unsigned int* peer_data, unsigned int* peer_inbox,
                              unsigned int* my_inbox, const unsigned int* src,
                              int words, int iters, int is_initiator, unsigned int* done)
{
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ int bail;
    if (!tid) bail = 0;
    __syncthreads();
    for (int i = 1; i <= iters; ++i) {
        if (!is_initiator) {
            if (!tid && !spin_until(my_inbox, i)) bail = i;
            __syncthreads();
            if (bail) break;
        }
        // posted writes: fire-and-forget, they pipeline across the link
        for (int k = tid; k < words; k += nt)
            __builtin_nontemporal_store(src[k], &peer_data[k]);
        __syncthreads();
        if (!tid) st_uncached(peer_inbox, i);
        if (is_initiator) {
            if (!tid && !spin_until(my_inbox, i)) bail = i;
            __syncthreads();
            if (bail) break;
        }
    }
    if (!tid) *done = bail;
}

// ---- PULL: barrier, then READ the peer's payload (what AITER 1-stage does) -------
__global__ void pingpong_pull(unsigned int* peer_data, unsigned int* peer_inbox,
                              unsigned int* my_inbox, unsigned int* dst,
                              int words, int iters, int is_initiator, unsigned int* done)
{
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ int bail;
    if (!tid) bail = 0;
    __syncthreads();
    for (int i = 1; i <= iters; ++i) {
        if (!tid) {
            st_uncached(peer_inbox, i);
            if (!spin_until(my_inbox, i)) bail = i;
        }
        __syncthreads();
        if (bail) break;
        // non-posted reads: each is a full round trip across the link
        unsigned int acc = 0;
        for (int k = tid; k < words; k += nt)
            acc += __builtin_nontemporal_load(&peer_data[k]);
        if (acc == 0xdeadbeefu) dst[0] = acc;   // keep the reads from being optimised out
        __syncthreads();
    }
    if (!tid) *done = bail;
}

int main()
{
    setvbuf(stdout, nullptr, _IOLBF, 0);   // survive a hang

    int n = 0;
    HIP_CHECK(hipGetDeviceCount(&n));
    printf("devices visible: %d\n", n);
    if (n < 2) { printf("need 2 GPUs\n"); return 1; }

    int can01 = 0, can10 = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can01, 0, 1));
    HIP_CHECK(hipDeviceCanAccessPeer(&can10, 1, 0));
    printf("peer capable: 0->1 %d   1->0 %d\n", can01, can10);
    if (!can01 || !can10) {
        printf("NO DIRECT P2P -- every peer access stages through host memory.\n"
               "That alone would explain 116 us, and the push rewrite is dead.\n");
        return 1;
    }
    HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipDeviceEnablePeerAccess(1, 0));
    HIP_CHECK(hipSetDevice(1)); HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));
    printf("peer access enabled both ways\n");

    const int WORDS = 8192 / 4;          // 8 KB = the real per-layer payload
    unsigned int *inbox[2], *data[2], *src[2], *dst[2], *done[2];
    hipStream_t st[2];
    for (int d = 0; d < 2; ++d) {
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipStreamCreate(&st[d]));
        HIP_CHECK(hipMalloc(&inbox[d], 256));
        HIP_CHECK(hipMalloc(&done[d], 256));
        HIP_CHECK(hipMalloc(&data[d], WORDS * 4));
        HIP_CHECK(hipMalloc(&src[d], WORDS * 4));
        HIP_CHECK(hipMalloc(&dst[d], WORDS * 4));
        HIP_CHECK(hipMemset(inbox[d], 0, 256));
        HIP_CHECK(hipMemset(done[d], 0, 256));
        HIP_CHECK(hipMemset(src[d], 1, WORDS * 4));
    }
    printf("buffers allocated\n");

    // ---------------- Test 0: peer write visibility (cannot hang) ----------------
    printf("\n--- test 0: does a peer write land? ---\n");
    for (int d = 0; d < 2; ++d) {
        int peer = 1 - d;
        unsigned int want = 0xABCDu + d;
        HIP_CHECK(hipSetDevice(peer)); HIP_CHECK(hipMemset(inbox[peer], 0, 4));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipSetDevice(d));
        hipLaunchKernelGGL(poke_peer, dim3(1), dim3(1), 0, 0, inbox[peer], want);
        HIP_CHECK(hipDeviceSynchronize());
        unsigned int got = 0;
        HIP_CHECK(hipSetDevice(peer));
        HIP_CHECK(hipMemcpy(&got, inbox[peer], 4, hipMemcpyDeviceToHost));
        printf("  dev%d -> dev%d : wrote 0x%X, peer reads 0x%X  %s\n",
               d, peer, want, got, (got == want) ? "OK" : "*** NOT VISIBLE ***");
        if (got != want) {
            printf("\nPeer writes do not land. Every spin below would deadlock;\n"
                   "stopping here rather than hanging the cards.\n");
            return 1;
        }
    }

    // ---------------- timed ping-pongs ----------------
    auto run = [&](const char* name, int iters, int kind) -> int {
        for (int d = 0; d < 2; ++d) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipMemset(inbox[d], 0, 256));
            HIP_CHECK(hipMemset(done[d], 0, 256));
            HIP_CHECK(hipDeviceSynchronize());
        }
        hipEvent_t a, b;
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
        HIP_CHECK(hipEventRecord(a, st[0]));
        for (int d = 0; d < 2; ++d) {
            HIP_CHECK(hipSetDevice(d));
            int init = (d == 0), peer = 1 - d;
            if (kind == 0)
                hipLaunchKernelGGL(pingpong_flag, dim3(1), dim3(1), 0, st[d],
                                   inbox[peer], inbox[d], iters, init, done[d]);
            else if (kind == 1)
                hipLaunchKernelGGL(pingpong_push, dim3(1), dim3(256), 0, st[d],
                                   data[peer], inbox[peer], inbox[d], src[d], WORDS, iters, init, done[d]);
            else
                hipLaunchKernelGGL(pingpong_pull, dim3(1), dim3(256), 0, st[d],
                                   data[peer], inbox[peer], inbox[d], dst[d], WORDS, iters, init, done[d]);
        }
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipEventRecord(b, st[0]));
        for (int d = 0; d < 2; ++d) { HIP_CHECK(hipSetDevice(d)); HIP_CHECK(hipStreamSynchronize(st[d])); }
        float ms = 0;
        HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipEventElapsedTime(&ms, a, b));
        unsigned int bail0 = 0, bail1 = 0;
        HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipMemcpy(&bail0, done[0], 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipSetDevice(1)); HIP_CHECK(hipMemcpy(&bail1, done[1], 4, hipMemcpyDeviceToHost));
        if (bail0 || bail1)
            printf("  %-12s BAILED at iter %u/%u -- spin never observed peer; timing invalid\n",
                   name, bail0 ? bail0 : bail1, iters);
        else
            printf("  %-12s %8.2f us per exchange   (%d iters, %.2f ms total)\n",
                   name, ms * 1000.0 / iters, iters, ms);
        return 0;
    };

    // ---------------- protocol matrix ----------------
    printf("\n--- which (store, load) pair is visible between RUNNING kernels? ---\n");
    {
        const char* ldn[] = {"glc+slc", "glc", "acquire"};
        const char* stn[] = {"glc+slc", "release"};
        auto probe = [&](int ld, int st_, auto kern) {
            for (int d = 0; d < 2; ++d) {
                HIP_CHECK(hipSetDevice(d));
                HIP_CHECK(hipMemset(inbox[d], 0, 256));
                HIP_CHECK(hipMemset(done[d], 0, 256));
                HIP_CHECK(hipDeviceSynchronize());
            }
            for (int d = 0; d < 2; ++d) {
                HIP_CHECK(hipSetDevice(d));
                hipLaunchKernelGGL(kern, dim3(1), dim3(1), 0, st[d],
                                   inbox[1 - d], inbox[d], 50, d == 0, done[d]);
            }
            for (int d = 0; d < 2; ++d) { HIP_CHECK(hipSetDevice(d)); HIP_CHECK(hipStreamSynchronize(st[d])); }
            unsigned int b0 = 0, b1 = 0;
            HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipMemcpy(&b0, done[0], 4, hipMemcpyDeviceToHost));
            HIP_CHECK(hipSetDevice(1)); HIP_CHECK(hipMemcpy(&b1, done[1], 4, hipMemcpyDeviceToHost));
            printf("  store=%-8s load=%-8s  %s\n", stn[st_], ldn[ld],
                   (b0 || b1) ? "NOT VISIBLE (stalled)" : "VISIBLE -- 50/50 round trips");
            return 0;
        };
        probe(0, 0, probe_protocol<0, 0>);
        probe(1, 0, probe_protocol<1, 0>);
        probe(2, 0, probe_protocol<2, 0>);
        probe(0, 1, probe_protocol<0, 1>);
        probe(1, 1, probe_protocol<1, 1>);
        probe(2, 1, probe_protocol<2, 1>);
    }

    printf("\npriming: loading code objects so no arm is timed against a cold launch\n");
    for (int d = 0; d < 2; ++d) {
        HIP_CHECK(hipSetDevice(d));
        int peer = 1 - d;
        hipLaunchKernelGGL(pingpong_flag, dim3(1), dim3(1), 0, st[d],
                           inbox[peer], inbox[d], 0, d == 0, done[d]);
        hipLaunchKernelGGL(pingpong_push, dim3(1), dim3(256), 0, st[d],
                           data[peer], inbox[peer], inbox[d], src[d], WORDS, 0, d == 0, done[d]);
        hipLaunchKernelGGL(pingpong_pull, dim3(1), dim3(256), 0, st[d],
                           data[peer], inbox[peer], inbox[d], dst[d], WORDS, 0, d == 0, done[d]);
    }
    for (int d = 0; d < 2; ++d) { HIP_CHECK(hipSetDevice(d)); HIP_CHECK(hipStreamSynchronize(st[d])); }
    printf("  primed\n");

    printf("\n--- timed, payload = %d B (one layer's all-reduce at batch 1) ---\n", WORDS * 4);
    printf("reference: NCCL costs 116 us per exchange in-model; 8 KB at 26 GB/s is 0.3 us\n");
    if (run("warmup", 100, 0)) return 1;
    if (run("rt_flag", 2000, 0)) return 1;
    if (run("rt_push_8k", 2000, 1)) return 1;
    if (run("rt_pull_8k", 2000, 2)) return 1;
    printf("\npush vs pull is the whole case for rewriting the kernel.\n");
    return 0;
}
