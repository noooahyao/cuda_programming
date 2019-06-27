#ifndef __INTRINSICS_CUH
#define __INTRINSICS_CUH

#include <cuda.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double *address, double val)
{
  unsigned long long int *address_as_ull =
      (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

template <typename T>
void printD(T *DeviceData, int n)
{
  T *tmp = new T[n];
  cudaMemcpy(tmp, DeviceData, n * sizeof(T), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < n; i++)
  {
    cout << tmp[i] << "\t";
    if (i%10==9)
    {
      cout<<endl;
    }
    
  }
}

// __device__ __forceinline__ void warpPrefixSum(int *val){

// }
// #if __CUDA_ARCH__ < 600
// __device__ double myatomicAdd(double* address, double val)
// {
//     unsigned long long int* address_as_ull =
//                               (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;

//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                         __double_as_longlong(val +
//                                __longlong_as_double(assumed)));

//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//     } while (assumed != old);

//     return __longlong_as_double(old);
// }
// #endif

__device__ int atomicAggInc(int *ctr)
{
  auto g = coalesced_threads();
  int warp_res;
  if (g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

__device__ __forceinline__ int warpReduceSum(int val)
{
  for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

__device__ __forceinline__ int blockReduceSum(int val)
{
  static __shared__ int shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);
  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
  if (wid == 0)
    val = warpReduceSum(val);
  return val;
}

#endif