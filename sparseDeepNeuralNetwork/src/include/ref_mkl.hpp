#include "common.cuh"
#include <mkl_spblas.h>



void infMkl(SparseMat *layers,
                      double bias,
                      SparseMat &featureVectors,
                      SparseMat &result,
                      SparseMat &tmp,
                      int nLayer,
                      int nSample,
                      int nNeuron)
{
    sparse_matrix_t csrA = NULL, csrB = NULL, csrC = NULL, csrD = NULL;
    mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, nSample, nNeuron, featureVectors.csrRowPtrA, featureVectors.csrRowPtrA + 1, featureVectors.csrColIndA, featureVectors.csrValA);
    mkl_sparse_optimize(csrA);

    ulong DNNedges = 0;

    Timer p;
    p.Start();

    int rows, cols, nnz;
    int rows2, cols2, nnz2;
    sparse_index_base_t indexing;
    sparse_index_base_t indexing2;

    mkl_sparse_d_create_csr(&csrB, SPARSE_INDEX_BASE_ZERO, nNeuron, nNeuron, layers[0].csrRowPtrA, layers[0].csrRowPtrA + 1, layers[0].csrColIndA, layers[0].csrValA);
    mkl_sparse_optimize(csrB);

    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrC);
    mkl_sparse_d_export_csr(csrC, &indexing, &rows, &cols, &result.csrRowPtrA, &result.csrRowPtrA + 1, &result.csrColIndA, &result.csrValA);
    nnz = result.csrRowPtrA[nSample];
    for (size_t i = 0; i < nnz; i++)
    {
        result.csrValA[0] = result.csrValA[0] + bias;
    }
    mkl_sparse_d_export_csr(csrC, &indexing, &rows, &cols, &result.csrRowPtrA, &result.csrRowPtrA + 1, &result.csrColIndA, &result.csrValA);
    mkl_sparse_optimize(csrC);

    for (size_t i = 1; i < nLayer; i++)
    {
        if (i % 2 == 1)
        {
            mkl_sparse_d_create_csr(&csrB, SPARSE_INDEX_BASE_ZERO, nNeuron, nNeuron, layers[i].csrRowPtrA, layers[i].csrRowPtrA + 1, layers[i].csrColIndA, layers[i].csrValA);
            mkl_sparse_optimize(csrB);

            mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, csrC, csrB, &csrD);
            mkl_sparse_d_export_csr(csrD, &indexing2, &rows2, &cols2, &tmp.csrRowPtrA, &tmp.csrRowPtrA + 1, &tmp.csrColIndA, &tmp.csrValA);
            nnz = tmp.csrRowPtrA[nSample];
            for (size_t j = 0; j < nnz; j++)
            {
                tmp.csrValA[j] = tmp.csrValA[j] + bias;
            }
        }
        else
        {
            mkl_sparse_d_create_csr(&csrB, SPARSE_INDEX_BASE_ZERO, nNeuron, nNeuron, layers[i].csrRowPtrA, layers[i].csrRowPtrA + 1, layers[i].csrColIndA, layers[i].csrValA);
            mkl_sparse_optimize(csrB);

            mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, csrD, csrB, &csrC);
            mkl_sparse_d_export_csr(csrC, &indexing, &rows, &cols, &result.csrRowPtrA, &result.csrRowPtrA + 1, &result.csrColIndA, &result.csrValA);
            nnz = result.csrRowPtrA[nSample];
            for (size_t j = 0; j < nnz; j++)
            {
                result.csrValA[j] = result.csrValA[j] + bias;
            }
        }
    }


    double t = p.Finish() / 1000;
    cout << "DNN neurons/layer: " << nNeuron << " , layers: " << nLayer << endl;
    cout << "Run time (sec): " << t << ", run rate (edges/sec): " << nSample * DNNedges / t << endl;


}
// };