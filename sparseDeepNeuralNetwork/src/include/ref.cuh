#include "common.cuh"
#include "intrinsics.cuh"
#include <cooperative_groups.h>
using namespace cooperative_groups;
// cg,rank
__global__ void csrmcsc_compute(int *csr_r,
                                int *csr_c,
                                double *csr_w,
                                int *csc_c,
                                int *csc_r,
                                double *csc_w,
                                int *out_r,
                                int *out_c,
                                double *out_w,
                                int *index,
                                int m,
                                int n,
                                int ldx,
                                double bias)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int laneid = threadIdx.x % 32;
    int warpid = tid / 32;
    if (warpid < m)
    {
        int row_len = csr_r[warpid + 1] - csr_r[warpid];
        if (row_len > 0)
        {
            for (size_t ii = laneid; ii < n; ii += 32)
            {
                // if ((csc_c[ii + 1] > csc_c[ii]) && (csr_c[csr_r[warpid + 1] - 1] > csc_r[csc_c[ii]]) && (csc_r[csc_c[ii + 1] - 1] > csr_c[csr_r[warpid]]))
                // {
                double local = 0;
                int kk = 0;
                for (size_t j = csc_c[ii]; j < csc_c[ii + 1]; j++)
                {
                    if (csc_r[j] > csr_c[csr_r[warpid] + kk])
                    {
                        while (csc_r[j] > csr_c[csr_r[warpid] + kk] && kk < row_len)
                            kk++;
                    }
                    if (csc_r[j] == csr_c[csr_r[warpid] + kk])
                        local += csc_w[j] * csr_w[csr_r[warpid] + kk];
                }
                local = max(local + bias, 0.0);
                if (local > 32)
                    local = 32;
                if (local != 0)
                {
                    int prev = atomicAggInc(&index[warpid]);
                    out_c[prev] = ii;
                    out_w[prev] = local;
                }
                // }
            }
        }
    }
}

__global__ void csrmcsc_compute2(int *csr_r,
                                 int *csr_c,
                                 double *csr_w,
                                 int *csc_c,
                                 int *csc_r,
                                 double *csc_w,
                                 int *out_r,
                                 int *out_c,
                                 double *out_w,
                                 int *index,
                                 int m,
                                 int n,
                                 int ldx,
                                 double bias,
                                 int *flag,
                                 int *nzrow)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int laneid = threadIdx.x % 32;
    int warpid = tid / 32;
    if (warpid < *nzrow)
    {
        int row_len = csr_r[flag[warpid] + 1] - csr_r[flag[warpid]];
        for (size_t ii = laneid; ii < n; ii += 32)
        {
            if (csc_c[ii + 1] > csc_c[ii])
            {
                double local = 0;
                int kk = 0;
                for (size_t j = csc_c[ii]; j < csc_c[ii + 1]; j++)
                {
                    if (csc_r[j] > csr_c[csr_r[flag[warpid]] + kk])
                    {
                        while (csc_r[j] > csr_c[csr_r[flag[warpid]] + kk] && kk < row_len)
                            kk++;
                    }
                    if (csc_r[j] == csr_c[csr_r[flag[warpid]] + kk])
                        local += csc_w[j] * csr_w[csr_r[flag[warpid]] + kk];
                }
                local = max(local + bias, 0.0);
                if (local > 32)
                    local = 32;
                if (local != 0)
                {
                    int prev = atomicAggInc(&index[flag[warpid]]);
                    out_c[prev] = ii;
                    out_w[prev] = local;
                }
            }
        }
    }
}

__global__ void filter(int *nnz_per_row,
                       int *flag,
                       int *nzrow,
                       int batch)
{
    // single warp function
    int tid = threadIdx.x;
    for (; tid < batch; tid += 32)
    {
        if ((nnz_per_row[tid + 1] != 0))
        {
            int prev = atomicAggInc(nzrow);
            flag[prev] = tid;
        }
    }
}
__global__ void csrmcsc_nnz(int *csr_r,
                            int *csr_c,
                            double *csr_w,
                            int *csc_c,
                            int *csc_r,
                            double *csc_w,
                            int *nnzPerRow,
                            int m,
                            int n,
                            int ldx,
                            double bias)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row_len = csr_r[blockIdx.x + 1] - csr_r[blockIdx.x];
    if (row_len > 0)
    {
        for (size_t ii = threadIdx.x; ii < n; ii += blockDim.x)
        {
            double local = 0;
            int kk = 0;
            for (size_t j = csc_c[ii]; j < csc_c[ii + 1]; j++)
            {
                if (csc_r[j] > csr_c[csr_r[blockIdx.x] + kk])
                {
                    while (csc_r[j] > csr_c[csr_r[blockIdx.x] + kk] && kk < row_len)
                        kk++;
                }
                if (csc_r[j] == csr_c[csr_r[blockIdx.x] + kk])
                {
                    local += csc_w[j] * csr_w[csr_r[blockIdx.x] + kk];
                    if (local > -bias)
                    {
                        atomicAdd(&nnzPerRow[blockIdx.x], 1);
                        break;
                    }
                }
            }
        }
    }
}

// each block process a row multiply a m to an dense vector then a csr?
// denvector too large?
template <int N>
__global__ void csrmcsc_to_dense(int *csr_r,
                                 int *csr_c,
                                 double *csr_w,
                                 int *csc_c,
                                 int *csc_r,
                                 double *csc_w,
                                 double *out,
                                 int m,
                                 int n,
                                 int ldx)
{
    __shared__ int tmp_row[N];
    __shared__ double tmp_w[N];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row_len = csr_r[blockIdx.x + 1] - csr_r[blockIdx.x];
    for (int tmp = threadIdx.x; tmp < row_len; tmp += blockDim.x)
    {
        tmp_row[tmp] = csr_c[csr_r[blockIdx.x] + tmp];
        tmp_w[tmp] = csr_w[csr_r[blockIdx.x] + tmp];
    }
    __syncthreads();

    if (row_len != 0)
    {
        for (size_t ii = threadIdx.x; ii < n; ii += blockDim.x)
        {
            int kk = 0;                                        //0
            for (size_t j = csc_c[ii]; j < csc_c[ii + 1]; j++) //j =1
            {
                if (csc_r[j] > tmp_row[kk]) // 1>2
                {
                    while (csc_r[j] > tmp_row[kk] && kk < row_len)
                    {
                        kk++;
                    }
                }
                if (csc_r[j] == tmp_row[kk])
                {
                    atomicAdd(&out[blockIdx.x + ii * ldx], csc_w[j] * tmp_w[kk]);
                }
            }
        }
    }
}

__global__ void csrmcsc_to_dense_nobuffer(int *csr_r,
                                          int *csr_c,
                                          double *csr_w,
                                          int *csc_c,
                                          int *csc_r,
                                          double *csc_w,
                                          double *out,
                                          int m,
                                          int n,
                                          int ldx)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row_len = csr_r[blockIdx.x + 1] - csr_r[blockIdx.x];
    if (row_len > 0)
    {
        for (size_t ii = threadIdx.x; ii < n; ii += blockDim.x)
        {
            int kk = 0;
            for (size_t j = csc_c[ii]; j < csc_c[ii + 1]; j++)
            {
                if (csc_r[j] > csr_c[csr_r[blockIdx.x] + kk])
                {
                    while (csc_r[j] > csr_c[csr_r[blockIdx.x] + kk] && kk < row_len)
                        kk++;
                }
                if (csc_r[j] == csr_c[csr_r[blockIdx.x] + kk])
                {
                    atomicAdd(&out[blockIdx.x + ii * ldx], csc_w[j] * csr_w[csr_r[blockIdx.x] + kk]);
                }
            }
        }
    }
}

__global__ void double_memset(double *data,
                              double val,
                              int m,
                              int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m)
    {
        for (size_t i = 0; i < n; i++)
        {
            data[m * i + tid] = val;
        }
    }
}
