#ifndef _MM_CUH__
#define _MM_CUH__

#include "matrix_io.cuh"
#include "sparse_conversion.cuh"
#include "safe_call_defs.h"
#include <cuda_runtime.h>
#include <cublas.h>

#include <cusparse.h>

double time_taken;
clock_t start, end;
int mm_repet = 1;



#define cudaCalloc(A, SIZE) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, SIZE); \
        if (__cudaCalloc_err == cudaSuccess) cudaMemset(*A, 0, SIZE); \
    } while (0)

void gpu_mm_sparse(struct Matrix *h_A, struct Matrix *h_B, struct Matrix *h_C, const int m, const int n, const int k,cusparseHandle_t handle,int time_flag,int repetation_flag) {
   cusparseOperation_t nop = CUSPARSE_OPERATION_NON_TRANSPOSE;

   struct SparseMat spmA,spmB,spmC;
   if(time_flag) start = clock();
   convert_to_sparse(&spmA, h_A, handle);
   convert_to_sparse(&spmB, h_B, handle);
   if(time_flag){
     end = clock();
     time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
     printf("Time for gpu_mm_sparse:conversion=%lf\n", time_taken);
     start = clock();
   }
   // init result sparse matrix
   cusparseCreateMatDescr(&(spmC.descr));
   cusparseSetMatType(spmC.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
   cusparseSetMatIndexBase(spmC.descr, CUSPARSE_INDEX_BASE_ZERO);
   spmC.num_rows = m;
   CudaSafeCall(cudaMalloc(&(spmC.csrRowPtrA_device),
                         (spmC.num_rows + 1) * sizeof(int)));

   cusparseSafeCall( cusparseXcsrgemmNnz(handle, nop, nop, m, n, k,
                spmA.descr,spmA.total_non_zero,spmA.csrRowPtrA_device,spmA.csrColIndA_device,
                spmB.descr,spmB.total_non_zero,spmB.csrRowPtrA_device,spmB.csrColIndA_device,
                spmC.descr, spmC.csrRowPtrA_device, &spmC.total_non_zero ));

  CudaSafeCall(cudaMalloc(&(spmC.csrColIndA_device),
                        spmC.total_non_zero * sizeof(int)));
  CudaSafeCall(cudaMalloc(&(spmC.csrValA_device),
                        spmC.total_non_zero * sizeof(float)));
   // Do the actual multiplication
   int i;
   if (repetation_flag){
     mm_repet = repetation_flag;
   }
   for(i=0;i<mm_repet;i++)
     cusparseSafeCall(cusparseDcsrgemm(handle, nop, nop, m, n, k,
                      spmA.descr,spmA.total_non_zero,spmA.csrValA_device,spmA.csrRowPtrA_device,spmA.csrColIndA_device,
                      spmB.descr,spmB.total_non_zero,spmB.csrValA_device,spmB.csrRowPtrA_device,spmB.csrColIndA_device,
                      spmC.descr,spmC.csrValA_device,spmC.csrRowPtrA_device,spmC.csrColIndA_device));
  if(time_flag){
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time for gpu_mm_sparse:cusparsemm=%lf\n", time_taken);
    start = clock();
  }
  convert_to_dense(&spmC, h_C, handle);
  h_C->is_column_first=1;
  if(time_flag){
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time for gpu_mm_sparse:back2dense=%lf\n", time_taken);
  }
   // Destroy the handle
  destroySparseMatrix(&spmA);
  destroySparseMatrix(&spmB);
  destroySparseMatrix(&spmC);
}

__global__ void sparseMM(int m, int n, float *a_csrVal, int *a_csrRowPtr, int *a_csrColInd,
   float *b_csrVal, int *b_csrRowPtr, int *b_csrColInd,
   float *c_matrix)
   /* blockDim.x-> n,  threadIdx.x-> j
      gridDim.x-> m, blockIdx.x-> i
      c_matrix -> m*n (columnbased)
      Each thread calculates i,j of matrix C.
      - To do that we look i th row of A
      - And j th row of B and add the matching values to the sum
   */
   {
     #ifdef DEBUG
       if(blockIdx.x==0 && threadIdx.x==1){
         printf("m:%d,n:%d\n",m,n);
       }
     #endif
     float sum = 0;
     int b_i,a_i,b_lim,a_lim,b_j,a_j;
     int id = blockIdx.x*blockDim.x+threadIdx.x;
     int c_i = id/n;
     int c_j = id%n;
     #ifdef DEBUG
       if(blockIdx.x==0 && threadIdx.x==1){
         printf("c_i:%d,c_j:%d,id:%d\n",c_i,c_j,id);
       }
     #endif
     if (c_i<m){
       id = index2DCol(c_i,c_j,m);
       a_lim = a_csrRowPtr[c_i+1];
       b_lim = b_csrRowPtr[c_j+1];
       a_i = a_csrRowPtr[c_i];
       b_i = b_csrRowPtr[c_j];
       #ifdef DEBUG
         if(blockIdx.x==0 && threadIdx.x==1){
           printf("Before:a_i=%d,b_i:%d,a_lim=%d,b_lim:%d\n",a_i,b_i,a_lim,b_lim);
         }
       #endif
       while((a_i<a_lim) && (b_i <b_lim))
          {
              b_j = b_csrColInd[b_i];
              a_j = a_csrColInd[a_i];
              #ifdef DEBUG
                if(blockIdx.x==0 && threadIdx.x==1){
                  printf("a_i=%d,b_i:%d,a_j=%d,b_j:%d\n",a_i,b_i,a_j,b_j);
                }
              #endif
              if ( a_j==b_j ){
                sum += a_csrVal[a_i]*b_csrVal[b_i];
                a_i++;
                b_i++;
                #ifdef DEBUG
                  if(blockIdx.x==0 && threadIdx.x==1){
                  printf("HIT:%f=%f*%f\n",a_csrVal[a_i]*b_csrVal[b_i],a_csrVal[a_i],b_csrVal[b_i]);
                  }
                #endif
              }
              else if (a_j<b_j){
                a_i++;
              }
              else{
                b_i++;
              }
          }
          #ifdef DEBUG
          if(blockIdx.x==0 && threadIdx.x==1){
            printf("sum:%f,id:%d\n",sum,id);
          }
          #endif
        c_matrix[id] = sum;
     }
   }

// void gpu_mm_sparse_ProjectImp(struct Matrix *h_A, struct Matrix *h_B, struct Matrix *h_C, const int m, const int n, const int k,cusparseHandle_t handle,int time_flag,int repetation_flag){
//   struct Matrix h_B_transposed;
//   struct SparseMat spmA,spmB;

//   if(time_flag) start = clock();
//   transpose2dMatrix(h_B,&h_B_transposed);
//   if(time_flag){
//     end = clock();
//     time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
//     printf("Time for gpu_mm_sparse_ProjectImp:transpose=%lf\n", time_taken);
//     start = clock();
//   }
//   convert_to_sparse(&spmA, h_A, handle);
//   convert_to_sparse(&spmB, &h_B_transposed, handle);
//   #ifdef DEBUG
//   copyDeviceCSR2Host(&spmA);
//   print_sparse_matrix(&spmA);
//   copyDeviceCSR2Host(&spmB);
//   print_sparse_matrix(&spmB);
//   #endif

//   //TODO check that m,n == h_C->dims[2] * h_C->dims[3]
//   int num_elems = m*n;
//   int nThreads = 1024;
//   int nBlocks = (num_elems / nThreads)+1;
//   float * matrix_device;
//   // Allocate device dense array
//   CudaSafeCall(cudaMalloc(&matrix_device,
//                         num_elems * sizeof(float)));
//   if(time_flag){
//     end = clock();
//     time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
//     printf("Time for gpu_mm_sparse_ProjectImp:conversion=%lf\n", time_taken);
//     start = clock();
//   }
//   int i;
//   if (repetation_flag){
//     mm_repet = repetation_flag;
//   }
//   for(i=0;i<mm_repet;i++)
//     sparseMM<<<nBlocks,nThreads>>>(m,n,spmA.csrValA_device,spmA.csrRowPtrA_device,spmA.csrColIndA_device,
//                       spmB.csrValA_device,spmB.csrRowPtrA_device,spmB.csrColIndA_device,matrix_device);

//   cudaDeviceSynchronize();
//   if(time_flag){
//     end = clock();
//     time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
//     printf("Time for gpu_mm_sparse_ProjectImp:sparsemm=%lf\n", time_taken);
//     start = clock();
//   }
//   CudaSafeCall(cudaMemcpy(h_C->vals,
//                           matrix_device,
//                           num_elems * sizeof(float),
//                           cudaMemcpyDeviceToHost));
//   if(time_flag){
//     end = clock();
//     time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
//     printf("Time for gpu_mm_sparse_ProjectImp:back2host=%lf\n", time_taken);
//     start = clock();
//   }
//   h_C->is_column_first=1;
//  cudaFree(matrix_device);
//  destroyMatrix(&h_B_transposed);
//  destroySparseMatrix(&spmA);
//  destroySparseMatrix(&spmB);
// }

// void cpu_mm(struct Matrix *h_A, struct Matrix *h_B, struct Matrix *h_C, int m, int n, int k){
//     int i,j,r;
//     float c_sum;
//     for(i=0;i<m;i++){
//       for(j=0;j<n;j++){
//         c_sum = 0;
//         for(r=0;r<k;r++){
//             c_sum += h_A->vals[index2DCol(i,r,m)]*h_B->vals[index2DCol(r,j,k)];
//         }
//         h_C->vals[index2DCol(i,j,m)] = c_sum;
//       }
//     }
//     h_C->is_column_first=1;
// }

#endif