// A complete 2-rank all-reduce staged through pinned host memory, timed against NCCL.
//
// WHY THIS SHAPE. Peer-VRAM flags do not work between these two MI210s (six memory
// ordering combinations, all stall intermittently). Pinned-host flags DO: 2000 round
// trips, no stalls, 4.49 us each. So put the barrier in host memory.
//
// But data-in-peer-VRAM + flag-in-host-memory take DIFFERENT PCIe paths, so nothing orders
// the flag after the data -- the peer could observe the flag and read stale staging. Fix:
// stage the DATA through host memory too. Same path, so PCIe ordering applies, and the
// flag written after the data is observed after it. This is what NCCL's SHM transport does
// for PCIe-only peers. Cost is one extra 8 KB hop: ~0.3 us at the 28 GB/s measured, which
// is noise next to a 4.5 us barrier.
//
// PROTOCOL (per rank, per all-reduce):
//   1. write my 8 KB input into my slot in host memory      (posted, ordered)
//   2. write my flag = seq                                   (lands after the data)
//   3. spin until peer's flag == seq
//   4. read peer's 8 KB slot, add to my input, write output
//
// REFERENCE: NCCL costs 116 us per 8 KB exchange in-model (10.46 ms / 90 per decode step).
// 8 KB at the link's 28 GB/s is 0.3 us, so NCCL is ~370x off bandwidth -- it is latency and
// protocol, not bytes. That is the gap this is trying to take.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

#define HC(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP error %s line %d\n", hipGetErrorString(e), __LINE__); return 1;} } while(0)

typedef unsigned int u32;
typedef float f32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ u32 ld_sys(const u32* p) {
  u32 v;
  asm volatile("global_load_dword %0, %1, off glc slc\n\ts_waitcnt vmcnt(0)"
               : "=v"(v) : "v"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_sys(u32* p, u32 v) {
  asm volatile("global_store_dword %0, %1, off glc slc\n\ts_waitcnt vmcnt(0)"
               : : "v"(p), "v"(v) : "memory");
}

#define TICKS 200000000ull
#define NTHREAD 256

// slot_me / slot_peer: 8 KB staging areas in host memory. flag_me / flag_peer likewise.
__global__ __launch_bounds__(NTHREAD) void allreduce_host(
    const f32x4* __restrict__ in, f32x4* __restrict__ out,
    f32x4* __restrict__ slot_me, const f32x4* __restrict__ slot_peer,
    u32* flag_me, const u32* flag_peer,
    int nvec, int iters, u32* done, unsigned int* ticks) {
  const int tid = threadIdx.x;
  __shared__ int bail;
  if (!tid) bail = 0;
  __syncthreads();

  for (int s = 1; s <= iters; ++s) {
    unsigned long long it0 = __builtin_amdgcn_s_memrealtime();
    // 1. publish my contribution into host staging
    for (int i = tid; i < nvec; i += NTHREAD)
      slot_me[i] = in[i];
    __syncthreads();                      // all lanes' stores issued
    // 2. flag, after the data, same path => ordered
    if (!tid) st_sys(flag_me, s);
    // 3. wait for the peer
    if (!tid) {
      unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
      while (ld_sys(flag_peer) < (u32)s)
        if (__builtin_amdgcn_s_memrealtime() - t0 > TICKS) { bail = s; break; }
    }
    __syncthreads();
    if (bail) break;
    // 4. reduce: mine + peer's
    for (int i = tid; i < nvec; i += NTHREAD) {
      f32x4 a = in[i], b = slot_peer[i];
      out[i] = a + b;
    }
    __syncthreads();
    if (!tid && ticks)
      ticks[s - 1] = (unsigned int)(__builtin_amdgcn_s_memrealtime() - it0);
  }
  if (!tid) *done = bail;
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int n = 0; HC(hipGetDeviceCount(&n));
  if (n < 2) { printf("need 2 GPUs\n"); return 1; }

  const int BYTES = 8192;                    // the real per-layer payload at batch 1
  const int NVEC = BYTES / sizeof(f32x4);

  // One host region: [slot0 8KB][slot1 8KB][flag0 pad][flag1 pad]
  const size_t HOSTSZ = 2 * BYTES + 8192;
  unsigned char* host = nullptr;
  HC(hipHostMalloc((void**)&host, HOSTSZ, hipHostMallocMapped | hipHostMallocCoherent));
  memset(host, 0, HOSTSZ);

  void* hdev[2];
  f32x4 *in[2], *out[2];
  u32* done[2];
  unsigned int* tick_d = nullptr;
  hipStream_t st[2];
  for (int d = 0; d < 2; ++d) {
    HC(hipSetDevice(d));
    HC(hipHostGetDevicePointer(&hdev[d], host, 0));
    HC(hipMalloc(&in[d], BYTES));
    HC(hipMalloc(&out[d], BYTES));
    HC(hipMemset(in[d], d + 1, BYTES));
    HC(hipMalloc(&done[d], 256));
    HC(hipMemset(done[d], 0, 256));
    HC(hipStreamCreate(&st[d]));
  }
  HC(hipSetDevice(0)); HC(hipMalloc(&tick_d, 4096 * sizeof(unsigned int)));
  printf("payload %d B, host staging mapped on both devices\n", BYTES);
  printf("reference: NCCL 116 us per 8 KB exchange in-model\n\n");

  {
    const int WARM = 100, iters = 2000;
    memset(host, 0, HOSTSZ);
    for (int d = 0; d < 2; ++d) {
      HC(hipSetDevice(d)); HC(hipMemset(done[d], 0, 256)); HC(hipDeviceSynchronize());
    }
    // --- 100 warm iterations, discarded ---
    for (int d = 0; d < 2; ++d) {
      HC(hipSetDevice(d));
      unsigned char* base = (unsigned char*)hdev[d];
      hipLaunchKernelGGL(allreduce_host, dim3(1), dim3(NTHREAD), 0, st[d],
                         in[d], out[d], (f32x4*)(base + d*BYTES), (f32x4*)(base + (1-d)*BYTES),
                         (u32*)(base + 2*BYTES + d*128), (u32*)(base + 2*BYTES + (1-d)*128),
                         NVEC, WARM, done[d], nullptr);
    }
    for (int d = 0; d < 2; ++d) { HC(hipSetDevice(d)); HC(hipStreamSynchronize(st[d])); }
    memset(host, 0, HOSTSZ);
    for (int d = 0; d < 2; ++d) { HC(hipSetDevice(d)); HC(hipMemset(done[d],0,256)); HC(hipDeviceSynchronize()); }

    hipEvent_t a, b;
    HC(hipSetDevice(0)); HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
    HC(hipEventRecord(a, st[0]));
    for (int d = 0; d < 2; ++d) {
      HC(hipSetDevice(d));
      unsigned char* base = (unsigned char*)hdev[d];
      f32x4* s_me   = (f32x4*)(base + d * BYTES);
      f32x4* s_peer = (f32x4*)(base + (1 - d) * BYTES);
      u32* f_me   = (u32*)(base + 2 * BYTES + d * 128);
      u32* f_peer = (u32*)(base + 2 * BYTES + (1 - d) * 128);
      hipLaunchKernelGGL(allreduce_host, dim3(1), dim3(NTHREAD), 0, st[d],
                         in[d], out[d], s_me, s_peer, f_me, f_peer, NVEC, iters, done[d], d == 0 ? tick_d : nullptr);
    }
    HC(hipSetDevice(0)); HC(hipEventRecord(b, st[0]));
    for (int d = 0; d < 2; ++d) { HC(hipSetDevice(d)); HC(hipStreamSynchronize(st[d])); }
    float ms = 0; HC(hipSetDevice(0)); HC(hipEventElapsedTime(&ms, a, b));
    u32 b0 = 0, b1 = 0;
    HC(hipSetDevice(0)); HC(hipMemcpy(&b0, done[0], 4, hipMemcpyDeviceToHost));
    HC(hipSetDevice(1)); HC(hipMemcpy(&b1, done[1], 4, hipMemcpyDeviceToHost));
    if (b0 || b1) {
      printf("  STALLED at iteration %u of %d -- host-memory handshake failed\n",
             b0 ? b0 : b1, iters);
      return 1;
    }
    double us = ms * 1000.0 / iters;
    printf("  mean %6.2f us per all-reduce   => %.2f ms for 90/step (NCCL: 14.45 ms)\n",
           us, us * 90 / 1000.0);
    std::vector<unsigned int> tk(iters);
    HC(hipSetDevice(0));
    HC(hipMemcpy(tk.data(), tick_d, iters * sizeof(unsigned int), hipMemcpyDeviceToHost));
    std::sort(tk.begin(), tk.end());
    double total_ticks = 0; for (auto t : tk) total_ticks += t;
    double tick_us = (ms * 1000.0) / total_ticks;      // self-calibrating counter rate
    auto q = [&](double f){ return tk[(size_t)(iters * f)] * tick_us; };
    printf("  per-iteration distribution (us):  p1 %.2f  p50 %.2f  p90 %.2f  p99 %.2f  max %.2f\n",
           q(0.01), q(0.50), q(0.90), q(0.99), tk.back() * tick_us);
    printf("  tail ratio p99/p50 = %.2fx   (a wide tail = stalls, not steady latency)\n",
           q(0.99) / q(0.50));
  }

  // correctness: every element must be in[0]+in[1]
  {
    unsigned char h0[8192], h1[8192];
    HC(hipSetDevice(0)); HC(hipMemcpy(h0, out[0], BYTES, hipMemcpyDeviceToHost));
    HC(hipSetDevice(1)); HC(hipMemcpy(h1, out[1], BYTES, hipMemcpyDeviceToHost));
    const f32x4* p0 = (const f32x4*)h0;
    f32x4 va, vb;
    unsigned char pa[16], pb[16];
    memset(pa, 1, 16); memset(pb, 2, 16);
    memcpy(&va, pa, 16); memcpy(&vb, pb, 16);
    f32x4 want = va + vb;
    int bad = 0;
    for (int i = 0; i < NVEC; ++i)
      for (int k = 0; k < 4; ++k)
        if (p0[i][k] != want[k]) { bad++; break; }
    printf("\ncorrectness: %s (%d/%d vectors wrong on rank0)\n",
           bad ? "FAIL" : "all elements == in0 + in1", bad, NVEC);
    printf("ranks agree: %s\n", memcmp(h0, h1, BYTES) == 0 ? "yes" : "NO");
  }
  return 0;
}
