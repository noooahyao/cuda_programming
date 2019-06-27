#include "common.cuh"
#include "graph.hpp"
#include "ref.cuh"
// #include "intrinsics.cuh"
#include <gflags/gflags.h>
#include <cusparse.h>
#include <algorithm>
#include <cub/cub.cuh>
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
        inputFile = "/home/pywang/graphchallenge/sparse-images-";
        categoryFile = "/home/pywang/graphchallenge/neuron";
        layerFile = "/home/pywang/graphchallenge/neuron";
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

    cudaSetDevice(0);
    cudaFree(0);

    csrgemm2Info_t info = NULL;
    cusparseHandle_t handle = 0;
    cusparseCreate(&handle);
    cusparseCreateCsrgemm2Info(&info);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    stringstream ss;
    ss << inputFile << nNeuronPerLayer << ".gr";
    string feature_name = ss.str();
    ss.str("");
    // ss.clear();
    cout << "read file start" << endl;
    Graph feature_vector(feature_name);
    feature_vector.Read();
    int nSample = 60000;
    int batch = nSample;

    int max_outdegree = 0;
    bool noBuffer = false;
    if (feature_vector.max_outdegree <= 1024)
        max_outdegree = 1024;
    else if (feature_vector.max_outdegree <= 2048)
        max_outdegree = 2048;
    else if (feature_vector.max_outdegree <= 4096)
        max_outdegree = 4096;
    else
        noBuffer = true;

    SparseMat featureVectors;
    featureVectors.csrRowPtrA = feature_vector.xadj;
    featureVectors.csrColIndA = feature_vector.adjncy;
    featureVectors.csrValA = feature_vector.adjwgt;
    featureVectors.total_non_zero = feature_vector.num_edges;
    featureVectors.num_rows = feature_vector.num_nodes;

    SparseCSC *layers = new SparseCSC[nLayer];

    for (int i = 0; i < nLayer; i++)
    {
        ss << layerFile << nNeuronPerLayer << "/n" << nNeuronPerLayer << "-l" << (i + 1) << ".gr";
        string layer_name = ss.str();
        ss.str("");
        Graph layer_vector(layer_name);
        // layer_vector.ReadGraphCSC();
        layer_vector.ReadCSC();
        layers[i].cscColPtrA = layer_vector.csc_xadj;
        layers[i].cscRowIndA = layer_vector.csc_adjncy;
        layers[i].cscValA = layer_vector.csc_adjwgt;
        layers[i].total_non_zero = layer_vector.num_edges;
        layers[i].num_cols = layer_vector.num_nodes;
    }

    int nNeuron = nNeuronPerLayer;
    malloccopyCSR2Device(&featureVectors);
    SparseMat &spmA = featureVectors;
    // SparseMat spmA[feature_vector.s_n];
    SparseCSC spmB;
    SparseMat spmC;
    // cout << "feature_vector finished" << endl;

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
    // cout << "tmp layer data memory usage: " << (double)((spmB.num_cols + 1 + spmB.total_non_zero) * sizeof(int) + spmB.total_non_zero * sizeof(double)) / 1024 / 1024 << " MB" << endl;

    double *buffer;
    CudaSafeCall(cudaMalloc(&buffer, batch * nNeuron * sizeof(double)));
    // cout << "required buffer size: " << (double)(batch * nNeuron * sizeof(double)) / 1024 / 1024 << " MB" << endl;

    // int *tmp_nnz;
    int tmp_nnz_h;
    int *tmp_nnz_per_row;
    // CudaSafeCall(cudaMalloc(&(tmp_nnz), sizeof(int)));
    CudaSafeCall(cudaMalloc(&(tmp_nnz_per_row), batch * sizeof(int)));
    // int *tmp_nnz_per_row_h=new int [batch];
    // cout << "1" << allnnz2 << endl;

    // long allnnz2 = 0;
    // for (size_t i = 0; i < feature_vector.s_n; i++)
    // {
    //     allnnz2 += spmA.total_non_zero;
    // }
    // cout << "feature non-zero item " << allnnz2 << endl;

    uint nstreams = 5;
    cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    for (int i = 0; i < nstreams; i++)
    {
        gpuErrorcheck(cudaStreamCreate(&(streams[i])));
    }

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cudaProfilerStart();
    Timer p;
    p.Start();
    for (size_t i = 0; i < nLayer; i++)
    {
        long allnnz = 0;
        copyCSC2D(&layers[i], &spmB);
        CudaSafeCall(cudaMemset(buffer, 0, batch * nNeuron * sizeof(double)));
        csrmcsc_to_dense_nobuffer<<<spmA.num_rows, 1024, 0, 0>>>(spmA.csrRowPtrA_device, spmA.csrColIndA_device, spmA.csrValA_device,
                                                                 spmB.cscColPtrA_device, spmB.cscRowIndA_device, spmB.cscValA_device,
                                                                 buffer, spmA.num_rows, nNeuron, batch);
        CudaSafeCall(cudaMemset(tmp_nnz_per_row, 0, batch * sizeof(int)));
        MatrixReluActive_nnz<<<spmA.num_rows / 1024 + 1, 1024>>>(buffer, spmA.num_rows, nNeuron, batch, bias, tmp_nnz_per_row);
        CudaSafeCall(cudaMemset(spmA.csrRowPtrA_device, 0, sizeof(int)));
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      tmp_nnz_per_row, spmA.csrRowPtrA_device + 1, spmA.num_rows);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      tmp_nnz_per_row, spmA.csrRowPtrA_device + 1, spmA.num_rows);
        CudaSafeCall(cudaMemcpy(&spmA.total_non_zero, spmA.csrRowPtrA_device + spmA.num_rows,
                                sizeof(int), cudaMemcpyDeviceToHost));

        if (spmA.total_non_zero > spmA.capacity)
        {
            reallocate(&spmA, spmA.total_non_zero);
        }
        allnnz += spmA.total_non_zero;

        MatrixToCSR<<<spmA.num_rows / 1024 + 1, 1024>>>(buffer, spmA.num_rows, nNeuron,
                                                        batch, bias, tmp_nnz_per_row,
                                                        spmA.csrRowPtrA_device,
                                                        spmA.csrColIndA_device,
                                                        spmA.csrValA_device);
        CudaSafeCall(cudaFree(d_temp_storage));
        // if (i % 50 == 0)
        // cout << "layer " << i << " compute finished with nnz\t" << allnnz << endl;
    }
    cudaProfilerStop();
    double t = p.Finish() / 1000;
    cout << "DNN neurons/layer: " << nNeuron << " , layers: " << nLayer << endl;
    cout << "Run time (sec): " << t << ", run rate (edges/sec): " << nSample * DNNedges / t << endl;

    vector<int> out_nnz;
    int nnz = 0;

    copyDeviceCSR2Host(&spmA);
    nnz += spmA.total_non_zero;

    Timer tr;
    tr.Start();

    for (size_t j = 0; j < spmA.num_rows; j++)
    {
        double tmp = 0;
        for (size_t k = spmA.csrRowPtrA[j]; k < spmA.csrRowPtrA[j + 1]; k++)
        {
            if (spmA.csrValA[k] > 0)
            {
                out_nnz.push_back(j);
                break;
            }
        }
    }
    // cout << "\nNumber of categories  " << out_nnz.size() << endl;
    // cout << " Identify the categories finished using " << tr.Finish() / 1000 << " s\n";
    // cout << "computed categories of first 10 \n";
    // for (size_t i = 0; i < 10; i++)
    // {
    //     cout << out_nnz[i] + 1 << "\t";
    // }
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

    // cout << "\ntrue categories of first 10 \n";
    // for (size_t i = 0; i < 10; i++)
    // {
    //     cout << ground_truth[i] << "\t";
    // }
    bool pass = true;
    for (size_t i = 0; i < ground_truth.size(); i++)
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
