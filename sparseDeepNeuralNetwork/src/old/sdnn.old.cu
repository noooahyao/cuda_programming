#include "common.cuh"
#include "graph.hpp"
#include "ref.cuh"
#include <cusparse.h>
#include <algorithm>
// #define cusparseSafeCall( err ) __cusparseSafeCall( err, __FILE__, __LINE__ )

int main(int argc, char const *argv[])
{
    string inputFile = "/home/pywang/graphchallenge/sparse-images-";
    string categoryFile = "/home/pywang/graphchallenge/neuron";
    string layerFile = "/home/pywang/graphchallenge/neuron";
    int nNeuronPerLayer = 1024;
    int nLayer = 120;
    double neuralNetBias[4] = {-0.3, -0.35, -0.4, -0.45};
    double bias = neuralNetBias[0];

    cudaSetDevice(0);
    cudaFree(0);

    csrgemm2Info_t info = NULL;
    cusparseHandle_t handle = 0;
    cusparseCreate(&handle);
    cusparseCreateCsrgemm2Info(&info);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    stringstream ss;
    ss << inputFile << nNeuronPerLayer << ".tsv";
    string feature_name = ss.str();
    ss.str("");
    // ss.clear();
    cout << "read file start" << endl;
    Graph feature_vector(feature_name);

    int batch = 1024;
    cout << "spilt file start" << endl;
    feature_vector.SplitReadGraph(batch);
    cout << "layer read " << feature_vector.s_n << endl;
    cout << "max_outdegree " << feature_vector.max_outdegree << endl;

    SparseMat featureVectors;
    featureVectors.csrRowPtrA = feature_vector.xadj;
    featureVectors.csrColIndA = feature_vector.adjncy;
    featureVectors.csrValA = feature_vector.adjwgt;
    featureVectors.total_non_zero = feature_vector.num_edges;
    featureVectors.num_rows = feature_vector.num_nodes;

    SparseCSC *layers = new SparseCSC[nLayer];

    int nSample = 60000;
    for (int i = 0; i < nLayer; i++)
    {
        ss << layerFile << nNeuronPerLayer << "/n" << nNeuronPerLayer << "-l" << (i + 1) << ".tsv";
        string layer_name = ss.str();
        ss.str("");
        // sss.clear();
        Graph layer_vector(layer_name);
        layer_vector.ReadGraphCSC();
        layers[i].cscColPtrA = layer_vector.csc_xadj;
        layers[i].cscRowIndA = layer_vector.csc_adjncy;
        layers[i].cscValA = layer_vector.csc_adjwgt;
        layers[i].total_non_zero = layer_vector.num_edges;
        layers[i].num_cols = layer_vector.num_nodes;
    }

    int nNeuron = nNeuronPerLayer;
    // malloccopyCSR2Device(&featureVectors);
    // SparseMat &spmA = featureVectors;
    SparseMat spmA[feature_vector.s_n];
    SparseCSC spmB;
    SparseMat spmC;

    for (size_t i = 0; i < feature_vector.s_n; i++)
    {
        mallocCopySpiltCSR2D(&feature_vector, &spmA[i], i);
    }
    cout << "feature_vector finished" << endl;

    // malloccopyCSR2Device(&spmA);

    long long DNNedges = 0;
    int bnnz = layers[0].total_non_zero;
    for (size_t i = 0; i < nLayer; i++)
    {
        DNNedges += layers[i].total_non_zero;
        if (bnnz < layers[i].total_non_zero)
        {
            bnnz = layers[i].total_non_zero;
        }
    }
    // spmB.capacity = bnnz;
    spmB.total_non_zero = bnnz;
    spmB.num_cols = nNeuron;
    mallocCSCDevice(&spmB);
    cout << "tmp layer data memory usage: " << (double)((spmB.num_cols + 1 + spmB.total_non_zero) * sizeof(int) + spmB.total_non_zero * sizeof(double)) / 1024 / 1024 << " MB" << endl;

    double *buffer;
    CudaSafeCall(cudaMalloc(&buffer, batch * nNeuron * sizeof(double)));

    // spmC.num_rows = batch;
    // spmC.total_non_zero = batch * nNeuron;
    // spmC.capacity = batch * nNeuron;
    // mallocCSRDevice(&spmC);

    // spmD.num_rows = batch;
    // spmD.total_non_zero = batch * nNeuron;
    // mallocCSRDevice(&spmD);

    cusparseMatDescr_t tmpdescr;
    cusparseCreateMatDescr(&(tmpdescr));
    cusparseSetMatType(tmpdescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(tmpdescr, CUSPARSE_INDEX_BASE_ZERO);

    Timer p;
    p.Start();

    int *tmp_nnz;
    int tmp_nnz_h;
    int *tmp_nnz_per_row;
    CudaSafeCall(cudaMalloc(&(tmp_nnz), sizeof(int)));
    CudaSafeCall(cudaMalloc(&(tmp_nnz_per_row), batch * sizeof(int)));

    long allnnz2=0;
    for (size_t i = 0; i < feature_vector.s_n; i++)
    {
        allnnz2+=spmA[i].total_non_zero;
    }
    cout<<"feature non-zero item "<<allnnz2<<endl;

    for (size_t i = 0; i < nLayer; i++)
    {
        long allnnz = 0;
        copyCSC2D(&layers[i], &spmB);
        for (size_t j = 0; j < feature_vector.s_n; j++)
        {
            // spmA[j]
            // CudaSafeCall(cudaMemset(buffer, 0, batch * nNeuron * sizeof(double)));
            double_memset<<<batch / 1024 + 1, 1024>>>(buffer, 0.0, batch, nNeuron);
            CudaSafeCall(cudaPeekAtLastError());
            CudaSafeCall(cudaDeviceSynchronize());
            csrmcsc_to_dense<<<spmA[j].num_rows, 1024, 1024 * (sizeof(int) + sizeof(double)), 0>>>(spmA[j].csrRowPtrA_device, spmA[j].csrColIndA_device, spmA[j].csrValA_device,
                                                                                                   spmB.cscColPtrA_device, spmB.cscRowIndA_device, spmB.cscValA_device,
                                                                                                   buffer, spmA[j].num_rows, nNeuron, batch);
            CudaSafeCall(cudaPeekAtLastError());
            CudaSafeCall(cudaDeviceSynchronize());

            matrixAddBias<<<spmA[j].num_rows / 512 + 1, 512>>>(buffer, spmA[j].num_rows, nNeuron, batch, bias);
            CudaSafeCall(cudaPeekAtLastError());
            CudaSafeCall(cudaDeviceSynchronize());

            CudaSafeCall(cudaMemset(tmp_nnz_per_row, 0, batch * sizeof(int)));
            CudaSafeCall(cudaMemset(tmp_nnz, 0, sizeof(int)));
            cusparseSafeCall(cusparseDnnz(handle,
                                          CUSPARSE_DIRECTION_ROW,
                                          //   spmA[j].num_rows,
                                          batch,
                                          nNeuron,
                                          tmpdescr,
                                          buffer,
                                          batch,
                                          tmp_nnz_per_row,
                                          tmp_nnz));
            CudaSafeCall(cudaPeekAtLastError());
            CudaSafeCall(cudaDeviceSynchronize());
            CudaSafeCall(cudaMemcpy(&tmp_nnz_h, tmp_nnz, sizeof(int), cudaMemcpyDeviceToHost));

            spmA[j].total_non_zero = tmp_nnz_h;
            if (tmp_nnz_h > 0)
            {
                allnnz += tmp_nnz_h;
            }
            if (tmp_nnz_h > spmA[j].capacity)
            {
                reallocate(&spmA[j], tmp_nnz_h);
            }
            cusparseSafeCall(cusparseDdense2csr(handle,
                                                spmA[j].num_rows,
                                                nNeuron,
                                                tmpdescr,
                                                buffer,
                                                batch,
                                                tmp_nnz_per_row,
                                                spmA[j].csrValA_device,
                                                spmA[j].csrRowPtrA_device,
                                                spmA[j].csrColIndA_device));
            CudaSafeCall(cudaPeekAtLastError());
            CudaSafeCall(cudaDeviceSynchronize());

            // remove zero
            // active<<<spmA[j].total_non_zero / 512 + 1, 512>>>(spmA[j].csrValA_device, bias, spmA[j].total_non_zero);
            // CudaSafeCall(cudaPeekAtLastError());
            // CudaSafeCall(cudaDeviceSynchronize());

            // cusparseSafeCall(cusparseDpruneCsr2csr_bufferSizeExt(handle,
            //                                                      int m,
            //                                                      int n,
            //                                                      int nnzA,
            //                                                      const cusparseMatDescr_t descrA,
            //                                                      const double *csrValA,
            //                                                      const int *csrRowPtrA,
            //                                                      const int *csrColIndA,
            //                                                      const double *threshold,
            //                                                      const cusparseMatDescr_t descrC,
            //                                                      const double *csrValC,
            //                                                      const int *csrRowPtrC,
            //                                                      const int *csrColIndC,
            //                                                      size_t *pBufferSizeInBytes));
            // cusparseSafeCall(cusparseDpruneCsr2csrNnz(handle,
            //                                           int m,
            //                                           int n,
            //                                           int nnzA,
            //                                           const cusparseMatDescr_t descrA,
            //                                           const double *csrValA,
            //                                           const int *csrRowPtrA,
            //                                           const int *csrColIndA,
            //                                           const double *threshold,
            //                                           const cusparseMatDescr_t descrC,
            //                                           int *csrRowPtrC,
            //                                           int *nnzTotalDevHostPtr,
            //                                           void *pBuffer));
            // cusparseSafeCall(cusparseDpruneCsr2csr(handle,
            //                                        int m,
            //                                        int n,
            //                                        int nnzA,
            //                                        const cusparseMatDescr_t descrA,
            //                                        const double *csrValA,
            //                                        const int *csrRowPtrA,
            //                                        const int *csrColIndA,
            //                                        const double *threshold,
            //                                        const cusparseMatDescr_t descrC,
            //                                        double *csrValC,
            //                                        const int *csrRowPtrC,
            //                                        int *csrColIndC,
            //                                        void *pBuffer));

            // cout << "compute part "<<j<<endl;;
        }
        if (allnnz)
        {
            cout << "nnz at layer " << i << "\t" << allnnz << endl;
        }
        // cout << "compute layer "<< i<<endl;

        if (i % 10 == 0)
            cout << "layer " << i << " compute finished" << endl;
    }

    double t = p.Finish() / 1000;
    cout << "DNN neurons/layer: " << nNeuron << " , layers: " << nLayer << endl;
    cout << "Run time (sec): " << t << ", run rate (edges/sec): " << nSample * DNNedges / t << endl;

    vector<int> out_nnz;
    int nnz = 0;

    for (size_t i = 0; i < feature_vector.s_n; i++)
    {
        copyDeviceCSR2Host(&spmA[i]);
        nnz += spmA[i].total_non_zero;
        for (size_t j = 0; j < spmA[i].num_rows; j++)
        {
            double tmp = 0;
            for (size_t k = spmA[i].csrRowPtrA[j]; k < spmA[i].csrRowPtrA[j + 1]; k++)
            {
                if (spmA[i].csrValA[k] > 0)
                    tmp += spmA[i].csrValA[k];
            }
            if (tmp > 0)
            {
                out_nnz.push_back(i * batch + j);
            }
        }
    }

    SparseMat result;

    ss.str("");
    ss << categoryFile << nNeuronPerLayer << "-l" << nLayer << "-categories.tsv";
    string true_files = ss.str();
    ifstream tf;
    tf.open(true_files);

    vector<int> ground_truth;
    string line;
    int val = 0;
    while (getline(tf, line))
    {
        ss.str("");
        ss.clear();
        ss << line;
        ss >> val;
        ground_truth.push_back(val);
    }

    cout << "true categories of first 10 \n";
    for (size_t i = 0; i < 10; i++)
    {
        cout << ground_truth[i] << "\t";
    }

    cout << "\n nnz " << nnz << endl;
    cout << "out_nnz.size " << out_nnz.size() << endl;

    cout << "\ncomputed categories of first 10 \n";
    for (size_t i = 0; i < 10; i++)
    {
        cout << out_nnz[i] << "\t";
    }

    cout << "\nmax element of first 10 \n";
    for (size_t i = 0; i < 10; i++)
    {
        cout << *std::max_element(&spmA[0].csrValA[spmA[0].csrRowPtrA[i]], &spmA[0].csrValA[spmA[0].csrRowPtrA[i + 1]]) << "\t";
    }

    return 0;
}
    // if (1)
    // {
    //     double *buffer;
    //     CudaSafeCall(cudaMalloc(&buffer, 1024 * 1024 * sizeof(double)));
    //     for (int k = 1024; k < 1025; k++)
    //     {

    //         int *csrr, *csrc, *cscc, *cscr;
    //         CudaSafeCall(cudaMallocManaged(&(csrr), (k + 1) * sizeof(int)));
    //         CudaSafeCall(cudaMallocManaged(&(csrc), k * k * sizeof(int)));
    //         CudaSafeCall(cudaMallocManaged(&(cscc), (k + 1) * sizeof(int)));
    //         CudaSafeCall(cudaMallocManaged(&(cscr), k * k * sizeof(int)));
    //         double *csrw, *cscw;
    //         CudaSafeCall(cudaMallocManaged(&(csrw), k * k * sizeof(double)));
    //         CudaSafeCall(cudaMallocManaged(&(cscw), k * k * sizeof(double)));
    //         for (size_t i = 0; i < k + 1; i++)
    //         {
    //             csrr[i] = i * k;
    //             cscc[i] = i * k;
    //         }
    //         for (size_t i = 0; i < k * k; i++)
    //         {
    //             csrc[i] = i % k;
    //             cscr[i] = i % k;
    //             csrw[i] = (double)(1);
    //             cscw[i] = (double)(1);
    //         }

    //         // csrr[0] = 0;
    //         // cout << "cscc: ";
    //         // for (size_t i = 0; i < k + 1; i++)
    //         // {
    //         //     cout << cscc[i] << "\t";
    //         // }

    //         // cout << "inited" << endl;
    //         CudaSafeCall(cudaMemset(buffer, 0, k * k * sizeof(double)));
    //         csrmcsc_to_dense<<<k, 1024, 1024 * (sizeof(int) + sizeof(double)), 0>>>(csrr, csrc, csrw,
    //                                                                                 cscc, cscr, cscw,
    //                                                                                 buffer, k, k, k);
    //         CudaSafeCall(cudaPeekAtLastError());
    //         CudaSafeCall(cudaDeviceSynchronize());

    //         double *buffer_h = new double[k * k];
    //         CudaSafeCall(cudaMemcpy(buffer_h, buffer, (k * k) * sizeof(double), cudaMemcpyDeviceToHost));

    //         for (size_t i = 0; i < k; i++)
    //         {
    //             cout << endl;
    //             for (size_t j = 0; j < k; j++)
    //             {
    //                 cout << buffer_h[i + j * k] << "\t";
    //             }
    //         }
    //     }
    //     return 0;
    // }