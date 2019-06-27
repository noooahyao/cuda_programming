#include "common.cuh"
#include "graph.hpp"
#include "ref_mkl.hpp"

int main(int argc, char const *argv[])
{
    string inputFile = "/home/pywang/graphchallenge/sparse-images-";
    string categoryFile = "/home/pywang/graphchallenge/neuron";
    string layerFile = "/home/pywang/graphchallenge/neuron";
    int nNeuronPerLayer = 1024;
    int nLayer = 120;

    double neuralNetBias[4] = {-0.3, -0.35, -0.4, -0.45};

    stringstream ss;
    ss << inputFile << nNeuronPerLayer << ".tsv";

    string feature_name = ss.str();
    ss.str("");

    Graph feature_vector(feature_name);
    feature_vector.Read();

    SparseMat feature_SM;
    feature_SM.csrRowPtrA = feature_vector.xadj;
    feature_SM.csrColIndA = feature_vector.adjncy;
    feature_SM.csrValA = feature_vector.adjwgt;
    feature_SM.total_non_zero = feature_vector.num_edges;
    feature_SM.num_rows = feature_vector.num_nodes;

    SparseMat *layer_SM = new SparseMat[nLayer];

    int nSample = 60000;

    for (int i = 0; i < nLayer; i++)
    {
        ss << layerFile << nNeuronPerLayer << "/n" << nNeuronPerLayer << "-l" << i + 1 << ".tsv";
        string layer_name = ss.str();
        ss.str("");
        Graph layer_vector(layer_name);
        layer_vector.Read();
        layer_SM[i].csrRowPtrA = layer_vector.xadj;
        layer_SM[i].csrColIndA = layer_vector.adjncy;
        layer_SM[i].csrValA = layer_vector.adjwgt;
        layer_SM[i].total_non_zero = layer_vector.num_edges;
        layer_SM[i].num_rows = layer_vector.num_nodes;
    }

    SparseMat result;
    result.total_non_zero = feature_SM.total_non_zero;
    result.num_rows = nSample;
    result.csrRowPtrA = new int(feature_SM.num_rows);
    result.csrColIndA = new int(feature_SM.total_non_zero);
    result.csrValA = new double(feature_SM.total_non_zero);
    SparseMat tmp;
    tmp.csrRowPtrA = new int(feature_SM.num_rows);
    tmp.csrColIndA = new int(feature_SM.total_non_zero);
    tmp.csrValA = new double(feature_SM.total_non_zero);

    infMkl(layer_SM, neuralNetBias[0], feature_SM, result, tmp, nLayer, nSample, nNeuronPerLayer);


    ss.str("");
    ss << categoryFile << nNeuronPerLayer<< "-l"<<nLayer << "-categories.tsv";
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
    if (ground_truth.size() != (sizeof(result.csrValA)/sizeof(result.csrValA[0])))
        cout << "Result is wrong!" << endl;
    else
    {   int i = 0;
        for (; i < ground_truth.size(); i++)
        {
            if (ground_truth[i] != result.csrValA[i])
            {
                break;
            }    
        }    
        if (i == ground_truth.size())
            cout << "Result is right!" << endl;
        else
            cout << "Result is wrong!" << endl;
        
    }
    
    return 0;
}
