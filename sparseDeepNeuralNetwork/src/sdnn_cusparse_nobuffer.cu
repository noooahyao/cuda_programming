#include "common.cuh"
#include "graph.hpp"
// #include "ref_cusparse.cuh"
#include "intrinsics.cuh"
#include <gflags/gflags.h>
#include <cusparse.h>
#include <algorithm>
// #include <cub/cub.cuh>
#include <cuda_profiler_api.h>
// #define cusparseSafeCall( err ) __cusparseSafeCall( err, __FILE__, __LINE__ )

DEFINE_string(input, "", "dataset location");
DEFINE_int32(nn, 1024, "Neural number per layer");
DEFINE_int32(nl, 120, "layer number");

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    string inputFile;
    string categoryFile;
    string layerFile;
    if (FLAGS_input == "")
    {
        inputFile = "/home/yyd/graph-challenge/sparse-images-";
        categoryFile = "/home/yyd/graph-challenge/neuron";
        layerFile = "/home/yyd/graph-challenge/neuron";
    }
    else
    {
        inputFile = FLAGS_input + "/sparse-images-";
        categoryFile = FLAGS_input + "/neuron";
        layerFile = FLAGS_input + "/neuron";
    }
    int nNeuronPerLayer = FLAGS_nn;
    int nLayer = FLAGS_nl;
    double neuralNetBias[4] = {-0.3, -0.35, -0.4, -0.45};
    double bias;
    switch (nNeuronPerLayer)
    {
    case 1024:
    {
        bias = neuralNetBias[0];
        break;
    }
    case 4096:
    {
        bias = neuralNetBias[1];
        break;
    }
    case 16384:
    {
        bias = neuralNetBias[2];
        break;
    }
    case 65536:
    {
        bias = neuralNetBias[3];
        break;
    }
    }

    stringstream ss;
    ss << inputFile << nNeuronPerLayer << ".gr";

    string feature_name = ss.str();
    ss.str("");

    int nSample = 60000;
    int batch = nSample;
    Graph feature_vector(feature_name);
    feature_vector.Read();

    SparseMat featureVectors;
    featureVectors.csrRowPtrA = feature_vector.xadj;
    featureVectors.csrColIndA = feature_vector.adjncy;
    featureVectors.csrValA = feature_vector.adjwgt;
    featureVectors.total_non_zero = feature_vector.num_edges;
    featureVectors.num_rows = feature_vector.num_nodes;

    SparseMat *layers = new SparseMat[nLayer];

    // int nSample = 60000;

    for (int i = 0; i < nLayer; i++)
    {
        ss << layerFile << nNeuronPerLayer << "/n" << nNeuronPerLayer << "-l" << i + 1 << ".gr";
        string layer_name = ss.str();
        ss.str("");
        Graph layer_vector(layer_name);
        layer_vector.Read();
        layers[i].csrRowPtrA = layer_vector.xadj;
        layers[i].csrColIndA = layer_vector.adjncy;
        layers[i].csrValA = layer_vector.adjwgt;
        layers[i].total_non_zero = layer_vector.num_edges;
        layers[i].num_rows = layer_vector.num_nodes;
        // cout<<"layer_SM[i].num_rows " <<layer_SM[i].num_rows<<endl;
    }

    cudaSetDevice(0);
    cudaFree(0);
    csrgemm2Info_t info = NULL;
    cusparseHandle_t handle = 0;
    cusparseCreate(&handle);
    cusparseCreateCsrgemm2Info(&info);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    int nNeuron = nNeuronPerLayer;
    malloccopyCSR2Device(&featureVectors);
    SparseMat &spmA = featureVectors;

    cusparseSafeCall(cusparseCreateMatDescr(&(spmA.descr)));
    cusparseSafeCall(cusparseSetMatType(spmA.descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseSafeCall(cusparseSetMatIndexBase(spmA.descr, CUSPARSE_INDEX_BASE_ZERO));

    SparseMat spmB;
    SparseMat spmC, spmE;
    cusparseMatDescr_t descrD;
    void *buffer = NULL;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    Timer p;
    p.Start();

    long long DNNedges = 0;

    cusparseSafeCall(cusparseCreateMatDescr(&(spmB.descr)));
    cusparseSafeCall(cusparseSetMatType(spmB.descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseSafeCall(cusparseSetMatIndexBase(spmB.descr, CUSPARSE_INDEX_BASE_ZERO));

    int bnnz = layers[0].total_non_zero;
    for (size_t i = 0; i < nLayer; i++)
    {
        DNNedges += layers[i].total_non_zero;
        if (bnnz < layers[i].total_non_zero)
        {
            bnnz = layers[i].total_non_zero;
        }
    }
    spmB.total_non_zero = bnnz;
    spmB.num_rows = nNeuron;
    mallocCSRDevice(&spmB);
    // cout << "tmp layer data memory usage: " << (double)((spmB.num_rows + 1 + spmB.total_non_zero) * sizeof(int) + spmB.total_non_zero * sizeof(double)) / 1024 / 1024 << " MB" << endl;
    copyCSR2D(&layers[0], &spmB);

    cusparseCreateMatDescr(&(spmC.descr));
    cusparseSetMatType(spmC.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(spmC.descr, CUSPARSE_INDEX_BASE_ZERO);
    spmC.num_rows = batch;
    spmC.total_non_zero = 1;
    spmC.capacity = 1;
    mallocCSRDevice(&spmC);

    int baseC, nnzC;
    int *nnzTotalDevHostPtr = &nnzC;
    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    for (size_t i = 0; i < nLayer; i++)
    {
        long allnnz = 0;
        copyCSR2D(&layers[i], &spmB);
 
        cudaFree(spmC.csrRowPtrA_device);
        cudaMalloc(&(spmC.csrRowPtrA_device), (spmA.num_rows + 1) * sizeof(int));
        cusparseSafeCall(cusparseXcsrgemmNnz(handle, transA,transB, batch, nNeuron, nNeuron,
                                              spmA.descr, spmA.total_non_zero, spmA.csrRowPtrA_device, spmA.csrColIndA_device,
					      spmB.descr, spmB.total_non_zero, spmB.csrRowPtrA_device, spmB.csrColIndA_device,
                                              spmC.descr, spmC.csrRowPtrA_device, nnzTotalDevHostPtr));
        //cout << " nnx" << endl;
        if (NULL != nnzTotalDevHostPtr)
            nnzC = *nnzTotalDevHostPtr;
        else
        {
            cudaMemcpy(&nnzC, spmC.csrRowPtrA_device + spmA.num_rows, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&baseC, spmC.csrRowPtrA_device, sizeof(int), cudaMemcpyDeviceToHost);
            nnzC -= baseC;
        }
        if (nnzC > 0)
        {
            allnnz += nnzC;
        }
        spmC.capacity = nnzC;
        spmC.total_non_zero = nnzC;
        CudaSafeCall(cudaFree(spmC.csrColIndA_device));
        CudaSafeCall(cudaFree(spmC.csrValA_device));
        CudaSafeCall(cudaMalloc(&(spmC.csrColIndA_device),
                                nnzC * sizeof(int)));
        CudaSafeCall(cudaMalloc(&(spmC.csrValA_device),
                                nnzC * sizeof(double)));
        cusparseSafeCall(cusparseDcsrgemm(handle, transA,transB,spmA.num_rows, nNeuron, nNeuron, 
                                           spmA.descr, spmA.total_non_zero, spmA.csrValA_device, spmA.csrRowPtrA_device, spmA.csrColIndA_device,
                                           spmB.descr, spmB.total_non_zero, spmB.csrValA_device, spmB.csrRowPtrA_device, spmB.csrColIndA_device,
                                           spmC.descr, spmC.csrValA_device, spmC.csrRowPtrA_device, spmC.csrColIndA_device));
        active<<<batch / 1024 + 1, 1024, 0, 0>>>(spmC.csrRowPtrA_device, spmC.csrValA_device, bias, batch);
        remove_zero_update(&spmA, &spmC, batch, nNeuron);

        // if (i % 10 == 0)
        //     cout << "layer: " << i << "  spmA.total_non_zero: " << spmA.total_non_zero << endl;
    }

    double t = p.Finish() / 1000;
    cout << "DNN neurons/layer: " << nNeuron << " , layers: " << nLayer << endl;
    cout << "Run time (sec): " << t << ", run rate (edges/sec): " << nSample * DNNedges / t << endl;
    // copyDeviceCSR2Host(&spmA[0]);

    vector<int> out_nnz;
    int nnz = 0;

    copyDeviceCSR2Host(&spmA);
    nnz += spmA.total_non_zero;
    int num = 0;
    for (size_t j = 0; j < spmA.num_rows; j++)
    {
        double tmp = 0;
        for (size_t k = spmA.csrRowPtrA[j]; k < spmA.csrRowPtrA[j + 1]; k++)
        {
            if (spmA.csrValA[k] > 0)
                tmp += spmA.csrValA[k];
        }
        if (tmp > 0)
        {
            out_nnz.push_back(j);
            num++;
        }
    }
    cout << "non zero row: " << num << endl;
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

    // cout << "true categories of first 10 \n";
    // for (size_t i = 0; i < 10; i++)
    // {
    //     cout << ground_truth[i] << "\t";
    // }
    // cout << "\nout_nnz.size " << out_nnz.size() << endl;

    // cout << "\ncomputed categories of first 10 \n";
    // for (size_t i = 0; i < 10; i++)
    // {
    //     cout << out_nnz[i] + 1 << "\t";
    // }
    bool pass = true;
    for (size_t i = 0; i < out_nnz.size(); i++)
    {
        if (ground_truth[i] != out_nnz[i] + 1)
        {
            pass = false;
            // cout << "At " << i << "ground_truth[i]:\t" << ground_truth[i] << "out_nnz[i]\t" << out_nnz[i] + 1 << endl;
        }
    }
    if (pass)
    {
        cout << "Challenge PASSED ";
    }
    else
        cout << "Challenge failed ";

    return 0;
}
