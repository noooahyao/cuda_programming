#ifndef _SPARSE_CONVERSION_CUH__
#define _SPARSE_CONVERSION_CUH__
/*
 * Header file for conversion between sparse and dense matrices
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * cuSPARSE assumes matrices are stored in column major order
 */
#include <math.h>
#include <cuda.h>
#include <cusparse.h>
#include "matrix_io.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include "matrix_io.cuh"
#include "safe_call_defs.h"
#define cusparseSafeCall(err) __cusparseSafeCall(err, __FILE__, __LINE__)

const double SMALL_NUM = 0.0000000001;
struct SparseMat
{
	cusparseMatDescr_t descr;
	int *csrRowPtrA;
	int *csrColIndA;
	double *csrValA;
	int *csrRowPtrA_device;
	int *csrColIndA_device;
	double *csrValA_device;
	int total_non_zero;
	int num_rows;
	int is_on_device;
	int capacity;
};
struct SparseCSC
{
	cusparseMatDescr_t descr;
	int *cscColPtrA;
	int *cscRowIndA;
	double *cscValA;
	int *cscColPtrA_device;
	int *cscRowIndA_device;
	double *cscValA_device;
	int total_non_zero;
	int num_cols;
	int is_on_device;
};
__global__ void categoryIdentify(int *csrr, double *csrw,
								 int *cat, int *nnz, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n)
	{
		int rowlen = csrr[tid + 1] - csrr[tid];
		for (size_t i = csrr[tid]; i < csrr[tid + 1]; i++)
		{
			if (csrw[i] > 0)
			{
				int t = atomicAdd(nnz, 1);
				cat[t] = tid;
				return;
			}
		}
	}
}

__global__ void matrixAddBias(double *buffer,
							  int m,
							  int n,
							  int ldx,
							  double bias)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < ldx)
	{
		for (size_t i = 0; i < n; i++)
		{
			if (buffer[i * ldx + tid] != 0)
			{
				buffer[i * ldx + tid] = max(buffer[i * m + tid] + bias, 0.0);
			}
		}
	}
}
// buffer is comlum-major
__global__ void MatrixReluActive_nnz(double *buffer,
									 int m,
									 int n,
									 int ldx,
									 double bias,
									 int *nnzPerRow)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < m)
	{
		for (size_t i = 0; i < n; i++)
		{
			double tm = buffer[i * ldx + tid];
			if (tm != 0)
			{
				double tmp = max(tm + bias, 0.0);
				if (tmp > 32)
				{
					tmp = 32;
				}
				if (tmp != 0)
				{
					atomicAdd(&nnzPerRow[tid], 1);
				}
				buffer[i * ldx + tid] = tmp;
			}
		}
	}
}
__global__ void MatrixToCSR(double *buffer,
							int m,
							int n,
							int ldx,
							double bias,
							int *nnzPerRow,
							int *csr_r,
							int *csr_c,
							double *csr_w)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < m)
	{
		if (nnzPerRow[tid] != 0)
		{
			// int ii = csr_r[tid];
			int idx = csr_r[tid];
			// if (idx - ii <= nnzPerRow[tid])
			// {
			for (size_t i = 0; i < n; i++)
			{
				if (buffer[i * ldx + tid] > 0)
				{
					csr_c[idx] = i;
					csr_w[idx] = buffer[i * ldx + tid];
					idx++;
				}
			}
			// }
		}
	}
}

__global__ void active(int *spm_row, double *spm,
					   double bias,
					   int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int i = spm_row[tid];
	if (tid < n)
	{
		while (i < spm_row[tid + 1])
		{
			spm[i] = min(max(spm[i] + bias, 0.0), 32.0);
			i++;
		}
	}
}
void remove_zero_update(SparseMat *to, SparseMat *from, int batch, int nNeuron)
{
	double threshold = 0;
	CudaSafeCall(cudaFree(to->csrRowPtrA_device));

	CudaSafeCall(cudaMalloc(&(to->csrRowPtrA_device),
							(to->num_rows + 1) * sizeof(int)));
	CudaSafeCall(cudaFree(to->csrColIndA_device));
	CudaSafeCall(cudaFree(to->csrValA_device));

	size_t lworkInBytes = 0;
	cusparseHandle_t handle = NULL;
	cusparseCreate(&handle);
	cudaStream_t stream = NULL;
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cusparseSetStream(handle, stream);
	cusparseDpruneCsr2csr_bufferSizeExt(
		handle,
		batch,
		nNeuron,
		from->total_non_zero,
		from->descr,
		from->csrValA_device,
		from->csrRowPtrA_device,
		from->csrColIndA_device,
		&threshold,
		to->descr,
		to->csrValA_device,
		to->csrRowPtrA_device,
		to->csrColIndA_device,
		&lworkInBytes);
	void *buffer;
	int nn = 0;
	lworkInBytes = 1024;
	cudaMalloc(&buffer, lworkInBytes);
	cusparseDpruneCsr2csrNnz(
		handle,
		batch,
		nNeuron,
		from->total_non_zero,
		from->descr,
		from->csrValA_device,
		from->csrRowPtrA_device,
		from->csrColIndA_device,
		&threshold,
		to->descr,
		to->csrRowPtrA_device,
		&nn, /* host */
		buffer);

	CudaSafeCall(cudaMalloc(&(to->csrColIndA_device),
							nn * sizeof(int)));
	CudaSafeCall(cudaMalloc(&(to->csrValA_device),
							nn * sizeof(double)));
	cusparseSafeCall(cusparseDpruneCsr2csr(
		handle,
		batch,
		nNeuron,
		from->total_non_zero,
		from->descr,
		from->csrValA_device,
		from->csrRowPtrA_device,
		from->csrColIndA_device,
		&threshold,
		to->descr,
		to->csrValA_device,
		to->csrRowPtrA_device,
		to->csrColIndA_device,
		buffer));
	to->total_non_zero = nn;
}

void mallocCSRDevice(struct SparseMat *spm)
{

	CudaSafeCall(cudaMalloc(&(spm->csrRowPtrA_device),
							(spm->num_rows + 1) * sizeof(int)));
	CudaSafeCall(cudaMalloc(&(spm->csrColIndA_device),
							spm->total_non_zero * sizeof(int)));
	CudaSafeCall(cudaMalloc(&(spm->csrValA_device), spm->total_non_zero * sizeof(double)));
}

void repoint(SparseMat *spmA, SparseMat *spmC)
{
	CudaSafeCall(cudaFree(spmA->csrRowPtrA_device));
	CudaSafeCall(cudaFree(spmA->csrColIndA_device));
	CudaSafeCall(cudaFree(spmA->csrValA_device));

	spmA->csrRowPtrA_device = spmC->csrRowPtrA_device;
	spmA->csrColIndA_device = spmC->csrColIndA_device;
	spmA->csrValA_device = spmC->csrValA_device;
	spmA->total_non_zero = spmC->total_non_zero;
	
	spmC->csrRowPtrA_device=nullptr;
	spmC->csrColIndA_device=nullptr;
	spmC->csrValA_device=nullptr;
}
void mallocCopySpiltCSR2D(Graph *graph, SparseMat *tmp, int i)
{
	int nnz = graph->s_nnz[i];
	int nv = graph->s_nv[i];
	tmp->num_rows = nv;
	tmp->total_non_zero = nnz;
	tmp->capacity = nnz;
	mallocCSRDevice(tmp);
	CudaSafeCall(cudaMemcpy(tmp->csrRowPtrA_device,
							graph->s_xadj[i],
							(nv + 1) * sizeof(int),
							cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(tmp->csrColIndA_device,
							graph->s_adjncy[i],
							nnz * sizeof(int),
							cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(tmp->csrValA_device,
							graph->s_adjwgt[i],
							nnz * sizeof(double),
							cudaMemcpyHostToDevice));
}

void copyCSR2D(SparseMat *spm_ptr, SparseMat *tmp)
{
	tmp->total_non_zero = spm_ptr->total_non_zero;
	CudaSafeCall(cudaMemcpy(tmp->csrRowPtrA_device,
							spm_ptr->csrRowPtrA,
							(spm_ptr->num_rows + 1) * sizeof(int),
							cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(tmp->csrColIndA_device,
							spm_ptr->csrColIndA,
							spm_ptr->total_non_zero * sizeof(int),
							cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(tmp->csrValA_device,
							spm_ptr->csrValA,
							spm_ptr->total_non_zero * sizeof(double),
							cudaMemcpyHostToDevice));
}

void copyCSC2D(SparseCSC *spm_ptr, SparseCSC *tmp)
{
	tmp->total_non_zero = spm_ptr->total_non_zero;
	CudaSafeCall(cudaMemcpy(tmp->cscColPtrA_device,
							spm_ptr->cscColPtrA,
							(spm_ptr->num_cols + 1) * sizeof(int),
							cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(tmp->cscRowIndA_device,
							spm_ptr->cscRowIndA,
							spm_ptr->total_non_zero * sizeof(int),
							cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(tmp->cscValA_device,
							spm_ptr->cscValA,
							spm_ptr->total_non_zero * sizeof(double),
							cudaMemcpyHostToDevice));
}

void mallocCSCDevice(SparseCSC *spm)
{
	CudaSafeCall(cudaMalloc(&(spm->cscColPtrA_device),
							(spm->num_cols + 1) * sizeof(int)));
	CudaSafeCall(cudaMalloc(&(spm->cscRowIndA_device),
							spm->total_non_zero * sizeof(int)));
	CudaSafeCall(cudaMalloc(&(spm->cscValA_device),
							spm->total_non_zero * sizeof(double)));
}

void convert_to_sparse(
	struct SparseMat *,
	struct Matrix *,
	cusparseHandle_t);
void convert_to_dense(
	struct SparseMat *,
	struct Matrix *,
	cusparseHandle_t);
void copyDeviceCSR2Host(struct SparseMat *);
void destroySparseMatrix(struct SparseMat *);
void print_sparse_matrix(struct SparseMat *);

void convert_to_sparse(struct SparseMat *spm,
					   struct Matrix *mat,
					   cusparseHandle_t handle)
{
	int *nz_per_row_device;
	double *matrix = mat->vals;
#ifdef DEBUG
	printf("[%d, %d, %d, %d]\n", mat->dims[0], mat->dims[1], mat->dims[2], mat->dims[3]);
#endif
	double *matrix_device;
	const int lda = mat->dims[2];
	spm->num_rows = mat->dims[2];
	spm->total_non_zero = 0;

	// Create cusparse matrix descriptors
	cusparseCreateMatDescr(&(spm->descr));
	cusparseSetMatType(spm->descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(spm->descr, CUSPARSE_INDEX_BASE_ZERO);

	// Allocate device dense array and copy over
	CudaSafeCall(cudaMalloc(&matrix_device,
							mat->dims[2] * mat->dims[3] * sizeof(double)));
	CudaSafeCall(cudaMemcpy(matrix_device,
							matrix,
							mat->dims[2] * mat->dims[3] * sizeof(double),
							cudaMemcpyHostToDevice));

	// Device side number of nonzero element per row of matrix
	CudaSafeCall(cudaMalloc(&(nz_per_row_device),
							spm->num_rows * sizeof(int)));
	cusparseSafeCall(cusparseDnnz(handle,
								  CUSPARSE_DIRECTION_ROW,
								  mat->dims[2],
								  mat->dims[3],
								  spm->descr,
								  matrix_device,
								  lda,
								  nz_per_row_device,
								  &(spm->total_non_zero)));

// Host side number of nonzero elements per row of matrix
#ifdef DEBUG
	printf("Num non zero elements: %d\n", spm->total_non_zero);
#endif

	// Allocate device sparse matrices
	CudaSafeCall(cudaMalloc(&(spm->csrRowPtrA_device),
							(spm->num_rows + 1) * sizeof(int)));
	CudaSafeCall(cudaMalloc(&(spm->csrColIndA_device),
							spm->total_non_zero * sizeof(int)));
	CudaSafeCall(cudaMalloc(&(spm->csrValA_device),
							spm->total_non_zero * sizeof(double)));

	// Call cusparse
	cusparseSafeCall(cusparseDdense2csr(
		handle,				 // cusparse handle
		mat->dims[2],		 // Number of rows
		mat->dims[3],		 // Number of cols
		spm->descr,			 // cusparse matrix descriptor
		matrix_device,		 // Matrix
		lda,				 // Leading dimension of the array
		nz_per_row_device,   // Non zero elements per row
		spm->csrValA_device, // Holds the matrix values
		spm->csrRowPtrA_device,
		spm->csrColIndA_device));

	cudaFree(matrix_device);
	cudaFree(nz_per_row_device);
	spm->is_on_device = 1;
#ifdef DEBUG
	printf("Converted matrix from dense to sparse\n");
#endif
}

void convert_to_dense(struct SparseMat *spm,
					  struct Matrix *mat,
					  cusparseHandle_t handle)
{
	/* Assumes that csrValA_device, csrRowPtrA_device, csrColIndA_device
   * all exist on the device and have been correctly populated
   * Also assumes the cusparse matrix descriptor in the spm has between
   * properly initialized
   */
	int num_elems = mat->dims[2] * mat->dims[3];
	double *matrix_device;
	const int lda = mat->dims[2];
// Allocate device dense array
#ifdef DEBUG
	printf("num_elems %d\n", num_elems);
#endif
	CudaSafeCall(cudaMalloc(&matrix_device,
							num_elems * sizeof(double)));

	cusparseSafeCall(cusparseDcsr2dense(handle,
										mat->dims[2],
										mat->dims[3],
										spm->descr,
										spm->csrValA_device,
										spm->csrRowPtrA_device,
										spm->csrColIndA_device,
										matrix_device,
										lda));

	// Copy matrix back to cpu and free device storage
	CudaSafeCall(cudaMemcpy(mat->vals,
							matrix_device,
							num_elems * sizeof(double),
							cudaMemcpyDeviceToHost));
	cudaFree(matrix_device);
}

void copyDeviceCSR2Host(struct SparseMat *spm_ptr)
{
	// Allocate host memory and copy device vals back to host
	// WARNING this may result in memory leak if you continously call this function
	spm_ptr->csrRowPtrA = (int *)calloc((spm_ptr->num_rows + 1), sizeof(int));
	spm_ptr->csrColIndA = (int *)calloc(spm_ptr->total_non_zero, sizeof(int));
	spm_ptr->csrValA = (double *)calloc(spm_ptr->total_non_zero, sizeof(double));
	CudaSafeCall(cudaMemcpy(spm_ptr->csrRowPtrA,
							spm_ptr->csrRowPtrA_device,
							(spm_ptr->num_rows + 1) * sizeof(int),
							cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(spm_ptr->csrColIndA,
							spm_ptr->csrColIndA_device,
							spm_ptr->total_non_zero * sizeof(int),
							cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(spm_ptr->csrValA,
							spm_ptr->csrValA_device,
							spm_ptr->total_non_zero * sizeof(double),
							cudaMemcpyDeviceToHost));
#ifdef DEBUG
	printf("Copied sparse matrix from device to host\n");
#endif
}

void copyCSR2Device(struct SparseMat *spm)
{
	CudaSafeCall(cudaMemcpy(spm->csrRowPtrA_device,
							spm->csrRowPtrA,
							(spm->num_rows + 1) * sizeof(int),
							cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(spm->csrColIndA_device,
							spm->csrColIndA,
							spm->total_non_zero * sizeof(int),
							cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(spm->csrValA_device,
							spm->csrValA,
							spm->total_non_zero * sizeof(double),
							cudaMemcpyHostToDevice));
}

void malloccopyCSR2Device(struct SparseMat *spm)
{

	CudaSafeCall(cudaMalloc(&(spm->csrRowPtrA_device),
							(spm->num_rows + 1) * sizeof(int)));

	CudaSafeCall(cudaMalloc(&(spm->csrColIndA_device),
							spm->total_non_zero * sizeof(int)));
	CudaSafeCall(cudaMalloc(&(spm->csrValA_device),
							spm->total_non_zero * sizeof(double)));

	CudaSafeCall(cudaMemcpy(spm->csrRowPtrA_device,
							spm->csrRowPtrA,
							(spm->num_rows + 1) * sizeof(int),
							cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(spm->csrColIndA_device,
							spm->csrColIndA,
							spm->total_non_zero * sizeof(int),
							cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(spm->csrValA_device,
							spm->csrValA,
							spm->total_non_zero * sizeof(double),
							cudaMemcpyHostToDevice));
}

void destroySparseMatrix(struct SparseMat *spm)
{
	cudaFree(spm->csrRowPtrA_device);
	cudaFree(spm->csrColIndA_device);
	cudaFree(spm->csrValA_device);
	// One should clean this separately, since it throws and error if the pointers are not init.
	// free(spm->csrRowPtrA);
	// free(spm->csrColIndA);
	// free(spm->csrValA);
}

void reallocate(struct SparseMat *spm, int n)
{
	spm->capacity = n;
	cudaFree(spm->csrColIndA_device);
	cudaFree(spm->csrValA_device);

	CudaSafeCall(cudaMalloc(&(spm->csrColIndA_device),
							n * sizeof(int)));
	CudaSafeCall(cudaMalloc(&(spm->csrValA_device),
							n * sizeof(double)));

	// One should clean this separately, since it throws and error if the pointers are not init.
	// free(spm->csrRowPtrA);
	// free(spm->csrColIndA);
	// free(spm->csrValA);
}

// void freeCol(struct SparseMat *spm,struct SparseMat *spm2){
//   // cudaFree(spm->csrRowPtrA_device);
//   cudaFree(spm->csrColIndA_device);
//   cudaFree(spm->csrValA_device);

// }
void check_size_swap(SparseMat *to, SparseMat *from)
{
	CudaSafeCall(cudaFree(to->csrColIndA_device));
	CudaSafeCall(cudaFree(to->csrValA_device));

	CudaSafeCall(cudaMalloc(&(to->csrColIndA_device),
							from->total_non_zero * sizeof(int)));
	CudaSafeCall(cudaMalloc(&(to->csrValA_device),
							from->total_non_zero * sizeof(double)));
	to->capacity = from->total_non_zero;

	CudaSafeCall(cudaFree(to->csrRowPtrA_device));

	CudaSafeCall(cudaMalloc(&(to->csrRowPtrA_device),
							(to->num_rows + 1) * sizeof(int)));
	CudaSafeCall(cudaMemcpy(to->csrRowPtrA_device,
							from->csrRowPtrA_device,
							(from->num_rows + 1) * sizeof(int),
							cudaMemcpyDeviceToDevice));
	CudaSafeCall(cudaMemcpy(to->csrColIndA_device,
							from->csrColIndA_device,
							from->total_non_zero * sizeof(int),
							cudaMemcpyDeviceToDevice));
	CudaSafeCall(cudaMemcpy(to->csrValA_device,
							from->csrValA_device,
							from->total_non_zero * sizeof(double),
							cudaMemcpyDeviceToDevice));
	to->total_non_zero = from->total_non_zero;
}
void print_sparse_matrix(struct SparseMat *spm)
{
	printf("Sparse representation of the matrix\n");
	int i, j, row_st, row_end, col;
	double num;
	for (i = 0; i < spm->num_rows; i++)
	{
		row_st = spm->csrRowPtrA[i];
		row_end = spm->csrRowPtrA[i + 1] - 1;
		for (j = row_st; j <= row_end; j++)
		{
			col = spm->csrColIndA[j];
			num = spm->csrValA[j];
			printf("(%d, %d): %05.2f\n", i, col, num);
		}
	}
}
#endif
