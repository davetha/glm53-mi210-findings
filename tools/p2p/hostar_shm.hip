// De-risk the one novel piece of the vLLM integration: can TWO SEPARATE PROCESSES share a
// POSIX shm arena, register it with HIP on different GPUs, and run the all-reduce handshake
// through it?
//
// Everything measured so far used ONE process with two devices (hipHostMalloc, mapped
// twice). vLLM runs TP workers as separate processes, so the staging arena must come from
// shm_open + mmap(MAP_SHARED) + hipHostRegister instead. That is the assumption the whole
// integration rests on, and it is untested.
//
// Two risks this checks:
//   1. Does hipHostRegister accept an shm-backed mapping at all on ROCm?
//   2. Do both processes' device pointers actually alias the same physical pages?
//
// Usage: hostar_shm <rank 0|1>   -- run both concurrently.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define HC(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  printf("[r%d] HIP error %s line %d\n", rank, hipGetErrorString(e), __LINE__); return 1;} } while(0)

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

#define TICKS 400000000ull      // ~4 s: generous, the peer process may start later
#define NT 256

// One exchange per launch -- the shape the real communicator needs, NOT a kernel that
// loops internally. `seq` lives in DEVICE memory and is incremented by the kernel, because
// under CUDA graphs the kernel arguments are baked at capture time: a host-side counter
// would replay the same value forever.
__global__ __launch_bounds__(NT) void hostar_once(
    const f32x4* __restrict__ in, f32x4* __restrict__ out,
    f32x4* __restrict__ slot_me, const f32x4* __restrict__ slot_peer,
    u32* flag_me, const u32* flag_peer, u32* seq_ctr, int nvec, u32* bail_out) {
  const int tid = threadIdx.x;
  __shared__ u32 seq;
  __shared__ int bail;
  if (!tid) { seq = ++(*seq_ctr); bail = 0; }
  __syncthreads();

  for (int i = tid; i < nvec; i += NT) slot_me[i] = in[i];
  __syncthreads();
  if (!tid) st_sys(flag_me, seq);
  if (!tid) {
    unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
    while (ld_sys(flag_peer) < seq)
      if (__builtin_amdgcn_s_memrealtime() - t0 > TICKS) { bail = 1; break; }
  }
  __syncthreads();
  if (!bail)
    for (int i = tid; i < nvec; i += NT) out[i] = in[i] + slot_peer[i];
  if (!tid) *bail_out = bail;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int rank = argc > 1 ? atoi(argv[1]) : 0;
  const char* NAME = "/vllm_hostar_test";
  const int BYTES = 8192;
  const int NVEC = BYTES / sizeof(f32x4);
  const size_t ARENA = 2 * BYTES + 8192;

  // rank 0 creates and sizes the segment; rank 1 waits for it to appear
  int fd = -1;
  if (rank == 0) {
    shm_unlink(NAME);
    fd = shm_open(NAME, O_CREAT | O_RDWR, 0600);
    if (fd < 0) { perror("shm_open create"); return 1; }
    if (ftruncate(fd, ARENA) != 0) { perror("ftruncate"); return 1; }
  } else {
    for (int i = 0; i < 200 && fd < 0; ++i) { fd = shm_open(NAME, O_RDWR, 0600); if (fd < 0) usleep(50000); }
    if (fd < 0) { perror("shm_open attach"); return 1; }
  }
  void* arena = mmap(nullptr, ARENA, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (arena == MAP_FAILED) { perror("mmap"); return 1; }
  if (rank == 0) memset(arena, 0, ARENA);
  printf("[r%d] shm mapped at %p\n", rank, arena);

  HC(hipSetDevice(rank));

  // RISK 1: will HIP register an shm-backed mapping?
  hipError_t reg = hipHostRegister(arena, ARENA, hipHostRegisterMapped);
  if (reg != hipSuccess) {
    printf("[r%d] *** hipHostRegister on shm FAILED: %s ***\n", rank, hipGetErrorString(reg));
    printf("[r%d] the POSIX-shm integration route is not viable as designed\n", rank);
    return 1;
  }
  void* adev = nullptr;
  HC(hipHostGetDevicePointer(&adev, arena, 0));
  printf("[r%d] registered OK, device pointer %p\n", rank, adev);

  f32x4 *in, *out; u32 *seq_ctr, *bail_d;
  HC(hipMalloc(&in, BYTES)); HC(hipMalloc(&out, BYTES));
  HC(hipMalloc(&seq_ctr, 4)); HC(hipMemset(seq_ctr, 0, 4));
  HC(hipMalloc(&bail_d, 4)); HC(hipMemset(bail_d, 0, 4));
  { // rank r contributes the value (r+1) in every lane
    float v = (float)(rank + 1);
    float* h = (float*)malloc(BYTES);
    for (int i = 0; i < BYTES / 4; ++i) h[i] = v;
    HC(hipMemcpy(in, h, BYTES, hipMemcpyHostToDevice)); free(h);
  }

  unsigned char* base = (unsigned char*)adev;
  f32x4* s_me   = (f32x4*)(base + rank * BYTES);
  f32x4* s_peer = (f32x4*)(base + (1 - rank) * BYTES);
  u32* f_me   = (u32*)(base + 2 * BYTES + rank * 128);
  u32* f_peer = (u32*)(base + 2 * BYTES + (1 - rank) * 128);

  // RISK 2: do the two processes' device pointers alias the same physical pages?
  const int ITERS = 2000, WARM = 100;
  hipEvent_t a, b; HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
  for (int i = 0; i < WARM; ++i)
    hipLaunchKernelGGL(hostar_once, dim3(1), dim3(NT), 0, 0,
                       in, out, s_me, s_peer, f_me, f_peer, seq_ctr, NVEC, bail_d);
  HC(hipDeviceSynchronize());
  u32 bail = 0; HC(hipMemcpy(&bail, bail_d, 4, hipMemcpyDeviceToHost));
  if (bail) { printf("[r%d] *** STALLED during warm-up -- peer never observed ***\n", rank); return 1; }

  HC(hipEventRecord(a));
  for (int i = 0; i < ITERS; ++i)
    hipLaunchKernelGGL(hostar_once, dim3(1), dim3(NT), 0, 0,
                       in, out, s_me, s_peer, f_me, f_peer, seq_ctr, NVEC, bail_d);
  HC(hipEventRecord(b));
  HC(hipEventSynchronize(b));
  float ms = 0; HC(hipEventElapsedTime(&ms, a, b));
  HC(hipMemcpy(&bail, bail_d, 4, hipMemcpyDeviceToHost));

  float res = 0;
  HC(hipMemcpy(&res, out, 4, hipMemcpyDeviceToHost));
  printf("[r%d] %s  %.2f us per all-reduce (per-launch, incl. dispatch)  out[0]=%.1f (want 3.0)\n",
         rank, bail ? "STALLED" : "OK", ms * 1000.0 / ITERS, res);

  HC(hipHostUnregister(arena));
  munmap(arena, ARENA);
  close(fd);
  if (rank == 0) shm_unlink(NAME);
  return 0;
}
