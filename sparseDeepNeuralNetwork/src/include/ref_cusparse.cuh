#include "common.cuh"
#include <cusparse.h>



template <typename T>
__global__ void printD(T *spm,
                       int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
    {
        printf("%d\t ", spm[tid]);
    }
}
template <typename T>
void printDevice(T *spm, int n)
{
    printD<<<1, 512>>>(spm, n);
}

void inferenceReLUvec(SparseMat *layers,
                      double bias,
                      SparseMat  featureVectors,
                      int nLayer,
                      int nSample,
                      int nNeuron)
{
    // layers [nLayer,nNeuron,nNeuron] 30%
    // bias [nLayer]
    // featureVectors [batchSize, nNeuron] 10%
    // Y = feature_batched;

    SparseMat &spmA = featureVectors;
    SparseMat spmB;
    SparseMat spmC, spmE;

    cusparseMatDescr_t descrD;
    csrgemm2Info_t info = NULL;
    cusparseCreateCsrgemm2Info(&info);
    void *buffer = NULL;
    cusparseHandle_t handle = 0;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    ulong DNNedges = 0;

    Timer p;
    p.Start();

    // CudaSafeCall(cudaMalloc(&(spmA.csrRowPtrA_device),
    //                         (spmA.num_rows + 1) * sizeof(int)));
    // CudaSafeCall(cudaMalloc(&(spmA.csrColIndA_device),
    //                         spmA.total_non_zero * sizeof(int)));
    // CudaSafeCall(cudaMalloc(&(spmA.csrValA_device),
    //                         spmA.total_non_zero * sizeof(double)));
    malloccopyCSR2Device(&spmA);
    // spmA.csrRowPtrA_device = featureVectors.csrRowPtrA_device;

    cout << "feature data memory usage: " << ((spmA.num_rows + 1 + spmA.total_non_zero) * sizeof(int) + spmA.total_non_zero * sizeof(double)) / 1024 / 1024 << " MB" << endl;

    // init layer storage
    cusparseCreateMatDescr(&(spmB.descr));
    cusparseSetMatType(spmB.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(spmB.descr, CUSPARSE_INDEX_BASE_ZERO);

    // todo
    spmB.total_non_zero = layers[0].total_non_zero;
    spmB.num_rows = nNeuron;

    CudaSafeCall(cudaMalloc(&(spmB.csrRowPtrA_device),
                            (spmB.num_rows + 1) * sizeof(int)));
    CudaSafeCall(cudaMalloc(&(spmB.csrColIndA_device),
                            spmB.total_non_zero * sizeof(int)));
    CudaSafeCall(cudaMalloc(&(spmB.csrValA_device),
                            spmB.total_non_zero * sizeof(double)));
    // alloccopyCSR2Device( &spmB);

    cout << "tmp layer data memory usage: " << (double)((spmB.num_rows + 1 + spmB.total_non_zero) * sizeof(int) + spmB.total_non_zero * sizeof(double)) / 1024 / 1024 << " MB" << endl;
    copyCSR2D(&layers[0], &spmB);

    // init result sparse matrix
    cusparseCreateMatDescr(&(spmC.descr));
    cusparseSetMatType(spmC.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(spmC.descr, CUSPARSE_INDEX_BASE_ZERO);
    spmC.num_rows = nSample;
    CudaSafeCall(cudaMalloc(&(spmC.csrRowPtrA_device),
                            (spmC.num_rows + 1) * sizeof(int)));
    // allocate suffcient memory
    CudaSafeCall(cudaMalloc(&(spmC.csrColIndA_device),
                            spmA.total_non_zero * sizeof(int)));
    CudaSafeCall(cudaMalloc(&(spmC.csrValA_device),
                            spmA.total_non_zero * sizeof(double)));

    // init result2 sparse matrix
    cusparseCreateMatDescr(&(spmE.descr));
    cusparseSetMatType(spmE.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(spmE.descr, CUSPARSE_INDEX_BASE_ZERO);
    spmE.num_rows = nSample;
    CudaSafeCall(cudaMalloc(&(spmE.csrRowPtrA_device),
                            (spmE.num_rows + 1) * sizeof(int)));
    // allocate suffcient memory
    CudaSafeCall(cudaMalloc(&(spmE.csrColIndA_device),
                            spmA.total_non_zero * sizeof(int)));
    CudaSafeCall(cudaMalloc(&(spmE.csrValA_device),
                            spmA.total_non_zero * sizeof(double)));

    size_t bufferSize_pre = spmA.total_non_zero;
    size_t bufferSize;
    cudaMalloc(&buffer, bufferSize_pre * sizeof(int));

    double alpha = 1.0;
    int baseC, nnzC;
    int *nnzTotalDevHostPtr = &nnzC;
    // stream layers to GPU and compute
    for (size_t i = 0; i < nLayer; i++)
    {

        // move layers to tmp_layer
        copyCSR2D(&layers[i], &spmB);

        malloccopyCSR2Device(&featureVectors);

        cout << "printDevice(featureVectors.csrRowPtrA_device)" << endl;
        printDevice<int>(featureVectors.csrRowPtrA_device, 10);

        cout << "printDevice(layers[i].csrRowPtrA_device)" << endl;
        printDevice<int>(layers[i].csrRowPtrA_device, 10);

        cout << "printDevice(spmA.csrRowPtrA_device)" << endl;
        printDevice<int>(spmA.csrRowPtrA_device, 10);
        cout << "printDevice(spmA.csrColIndA_device)" << endl;
        printDevice<int>(spmA.csrColIndA_device, 10);
        cout << "printDevice(spmB.csrRowPtrA_device)" << endl;
        printDevice<int>(spmB.csrRowPtrA_device, 10);
        cout << "printDevice(spmB.csrColIndA_device)" << endl;
        printDevice<int>(spmB.csrColIndA_device, 10);

        cusparseSafeCall(cusparseDcsrgemm2_bufferSizeExt(handle, nSample, nNeuron, nNeuron, &alpha,
                                                         spmA.descr, spmA.total_non_zero, spmA.csrRowPtrA_device, spmA.csrColIndA_device,
                                                         spmB.descr, spmB.total_non_zero, spmB.csrRowPtrA_device, spmB.csrColIndA_device,
                                                         nullptr,
                                                         descrD, 0, nullptr, nullptr,
                                                         info,
                                                         &bufferSize));

        if (bufferSize_pre < bufferSize)
        {
            CudaSafeCall(cudaFree(buffer));
            bufferSize_pre = bufferSize;
            CudaSafeCall(cudaMalloc(&buffer, bufferSize_pre));
        }
        cusparseSafeCall(cusparseXcsrgemm2Nnz(handle, nSample, nNeuron, nNeuron,
                                              spmA.descr, spmA.total_non_zero, spmA.csrRowPtrA_device, spmA.csrColIndA_device,
                                              spmB.descr, spmB.total_non_zero, spmB.csrRowPtrA_device, spmB.csrColIndA_device,
                                              descrD, 0, nullptr, nullptr,
                                              spmC.descr, spmC.csrRowPtrA_device, &spmC.total_non_zero,
                                              info, buffer));
        // if (NULL != nnzTotalDevHostPtr)
        // {
        //     nnzC = *nnzTotalDevHostPtr;
        // }
        // else
        // {
        //     cudaMemcpy(&nnzC, spmC.csrRowPtrA_device + nSample, sizeof(int), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(&baseC, spmC.csrRowPtrA_device, sizeof(int), cudaMemcpyDeviceToHost);
        //     nnzC -= baseC;
        // }

        cout << "spmA.total_non_zero: " << spmA.total_non_zero << endl;
        cout << "spmC.total_non_zero: " << spmC.total_non_zero << endl;
        if (nnzC > spmA.total_non_zero)
        {
            CudaSafeCall(cudaFree(spmC.csrColIndA_device));
            CudaSafeCall(cudaFree(spmC.csrValA_device));
            CudaSafeCall(cudaMalloc(&(spmC.csrColIndA_device),
                                    nnzC * sizeof(int)));
            CudaSafeCall(cudaMalloc(&(spmC.csrValA_device),
                                    nnzC * sizeof(double)));
        }

        // actually compute
        cusparseSafeCall(cusparseDcsrgemm2(handle, nSample, nNeuron, nNeuron, &alpha,
                                           spmA.descr, spmA.total_non_zero, spmA.csrValA_device, spmA.csrRowPtrA_device, spmA.csrColIndA_device,
                                           spmB.descr, spmB.total_non_zero, spmB.csrValA_device, spmB.csrRowPtrA_device, spmB.csrColIndA_device,
                                           nullptr,
                                           descrD, 0, nullptr, nullptr, nullptr,
                                           spmC.descr, spmC.csrValA_device, spmC.csrRowPtrA_device, spmC.csrColIndA_device,
                                           info, &bufferSize));
        // GPUmemcpy(spmC,spmE);
        // logical(spmE);
        active<<<spmA.total_non_zero / 512, 512>>>(spmA, bias, spmA.total_non_zero);
        cout << "cmpute once " << endl;
    }

    double t = p.Finish() / 1000;
    cout << "DNN neurons/layer: " << nNeuron << " , layers: " << nLayer << endl;
    cout << "Run time (sec): " << t << ", run rate (edges/sec): " << nSample * DNNedges / t << endl;
    copyDeviceCSR2Host(&spmA);
}
// };