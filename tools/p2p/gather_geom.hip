// Does the gather's launch geometry actually cost bandwidth? Variance-free, kernel-level.
//
// MEASURED IN-MODEL: 433 MB/step in 23.75 ms = 18.2 GB/s. This link delivers 23.9 GB/s to
// both sequential AND scattered 8 MiB reads (scatter_bw.py), so the access pattern is not
// the explanation -- there is ~5.6 ms/step unaccounted for.
//
// The ledger swept CHUNKS/LANES twice and called it flat, but measured END-TO-END ms/step
// on a box whose launch-to-launch drift is +/-7% (~6 ms). A 5.6 ms effect is invisible
// there, so "flat" is plausibly a false negative -- the same trap that sank the all-reduce
// work. This measures the KERNEL, where there is no launch noise.
//
// MECHANISM UNDER TEST: grid is (chunks, lanes) and `lanes` indexes MISSES:
//     for (int j = blockIdx.y; j < nm; j += gridDim.y)
// At the real nm ~= 1.28, lanes 2..63 retire immediately, so only ~chunks blocks copy:
// 16 x 256 = 4096 threads x 16 B = ~64 KB in flight. Covering 24 GB/s at ~1.5 us PCIe
// latency needs ~36 KB, so the default sits right at the edge -- raising `chunks` should
// raise achieved bandwidth if this is the cause, and do nothing if it is not.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cstring>
#include <cstdlib>

typedef unsigned int u32;
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
#define NT 256

#define HC(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP error %s line %d\n", hipGetErrorString(e), __LINE__); return 1;} } while(0)

// --- verbatim from vllm-expert-cache/kernels/expert_cache.hip ---
extern "C" __global__ __launch_bounds__(NT) void expert_cache_gather_k(
    u32x4* __restrict__ d0, const u32x4* __restrict__ s0, int v0,
    u32x4* __restrict__ d1, const u32x4* __restrict__ s1, int v1,
    u32x4* __restrict__ d2, const u32x4* __restrict__ s2, int v2,
    u32x4* __restrict__ d3, const u32x4* __restrict__ s3, int v3,
    u32x4* __restrict__ d4, const u32x4* __restrict__ s4, int v4,
    u32x4* __restrict__ d5, const u32x4* __restrict__ s5, int v5,
    const int* __restrict__ miss, const int* __restrict__ n_miss) {
  const int nm = n_miss[0];
  const int stride = gridDim.x * NT;
  const int start = blockIdx.x * NT + threadIdx.x;
  for (int j = blockIdx.y; j < nm; j += gridDim.y) {
    const int e = miss[2 * j];
    const int sl = miss[2 * j + 1];
#define CP(D, S, V)                                                                    \
  {                                                                                    \
    const u32x4* __restrict__ p = (S) + (size_t)e * (size_t)(V);                       \
    u32x4* __restrict__ q = (D) + (size_t)sl * (size_t)(V);                            \
    for (int i = start; i < (V); i += stride) q[i] = p[i];                             \
  }
    CP(d0, s0, v0) CP(d1, s1, v1) CP(d2, s2, v2)
    CP(d3, s3, v3) CP(d4, s4, v4) CP(d5, s5, v5)
#undef CP
  }
}

// Real per-expert tensor sizes at moe_intermediate_size=2048, hidden=4096, int4 g128.
// w13 packed 2*2048*4096/2 = 8 MiB ; w13 scale 2*2048*32*2 = 256 KiB
// w2  packed 4096*2048/2   = 4 MiB ; w2  scale 4096*16*2   = 128 KiB   => 12.375 MiB
static const long BYTES[6] = {
    8L << 20, 256L << 10, 4L << 20, 128L << 10, 16, 16,
};
static int N_EXPERT = 48;                // overridable: argv[1]

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc > 1) N_EXPERT = atoi(argv[1]);
  long per_expert = 0;
  for (int i = 0; i < 6; i++) per_expert += BYTES[i];
  printf("per-expert %.3f MiB, %d experts pinned = %.1f MiB\n",
         per_expert / 1048576.0, N_EXPERT, N_EXPERT * per_expert / 1048576.0);

  void *hs[6], *ds[6], *hs_dev[6];
  for (int i = 0; i < 6; i++) {
    HC(hipHostMalloc(&hs[i], (size_t)BYTES[i] * N_EXPERT, hipHostMallocMapped));
    HC(hipHostGetDevicePointer(&hs_dev[i], hs[i], 0));     // UVA view, as the offloader uses
    HC(hipMalloc(&ds[i], (size_t)BYTES[i] * 8));           // 8 slots
    memset(hs[i], i + 1, (size_t)BYTES[i] * N_EXPERT);
  }

  int *miss_d, *nmiss_d;
  HC(hipMalloc(&miss_d, 2 * 64 * sizeof(int)));
  HC(hipMalloc(&nmiss_d, sizeof(int)));

  printf("\n%-8s %-7s %-5s %10s %9s %9s   %s\n",
         "chunks", "lanes", "nm", "blocks", "ms", "GB/s", "vs in-model 18.2");

  // nm=1 and 2 bracket the measured 1.28 misses/layer-step; 4 shows the trend.
  for (int nm : {1, 2}) {
    std::vector<int> miss(2 * nm);
    for (int j = 0; j < nm; j++) { miss[2*j] = (j * 7) % N_EXPERT; miss[2*j+1] = j % 8; }
    HC(hipMemcpy(miss_d, miss.data(), miss.size()*sizeof(int), hipMemcpyHostToDevice));
    HC(hipMemcpy(nmiss_d, &nm, sizeof(int), hipMemcpyHostToDevice));

    for (int chunks : {16, 32}) {
      int lanes = 64;                                   // production default
      dim3 g(chunks, lanes), b(NT);
      auto run = [&]{
        hipLaunchKernelGGL(expert_cache_gather_k, g, b, 0, 0,
          (u32x4*)ds[0], (const u32x4*)hs_dev[0], (int)(BYTES[0]/16),
          (u32x4*)ds[1], (const u32x4*)hs_dev[1], (int)(BYTES[1]/16),
          (u32x4*)ds[2], (const u32x4*)hs_dev[2], (int)(BYTES[2]/16),
          (u32x4*)ds[3], (const u32x4*)hs_dev[3], (int)(BYTES[3]/16),
          (u32x4*)ds[4], (const u32x4*)hs_dev[4], (int)(BYTES[4]/16),
          (u32x4*)ds[5], (const u32x4*)hs_dev[5], (int)(BYTES[5]/16), miss_d, nmiss_d);
      };
      for (int i = 0; i < 5; i++) run();
      HC(hipDeviceSynchronize());
      hipEvent_t a, bb; HC(hipEventCreate(&a)); HC(hipEventCreate(&bb));
      const int IT = 50;
      HC(hipEventRecord(a));
      for (int i = 0; i < IT; i++) run();
      HC(hipEventRecord(bb));
      HC(hipEventSynchronize(bb));
      float ms = 0; HC(hipEventElapsedTime(&ms, a, bb)); ms /= IT;
      double gbs = (double)per_expert * nm / 1e9 / (ms / 1e3);
      int active = chunks * (nm < lanes ? nm : lanes);
      printf("%-8d %-7d %-5d %10d %9.3f %9.2f   %s\n", chunks, lanes, nm, active, ms, gbs,
             chunks == 16 ? "<- PRODUCTION DEFAULT" : "");
    }
    printf("\n");
  }
  printf("link delivers 23.9 GB/s to both sequential and scattered 8 MiB reads.\n");
  printf("If GB/s rises with chunks, the ledger's e2e 'geometry is flat' was a false negative.\n");
  return 0;
}
