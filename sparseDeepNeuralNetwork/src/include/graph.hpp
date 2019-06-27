#ifndef GRAPH_HPP
#define GRAPH_HPP
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
// using namespace std;
// #include "graph.hpp"
#include <cassert>
// #include "common.cuh"
struct Edge
{
	int source;
	int end;
};

class Graph
{
private:
public:
	std::string graphFilePath;

	bool hasZeroID;
	int num_nodes;
	int64_t num_edges;
	std::vector<Edge> edges;
	std::vector<double> weights;
	int64_t sizeEdgeTy;

	int *xadj, *vwgt, *adjncy;
	double *adjwgt;
	int *inDegree;
	int *outDegree;

	int *csc_xadj, *csc_vwgt, *csc_adjncy;
	double *csc_adjwgt;

	bool weighted;

	int **s_xadj, **s_adjncy;
	double **s_adjwgt;
	int **s_outDegree;
	int *s_nnz, *s_nv;
	int s_n;
	bool spilted;

	int max_outdegree;

	Graph(std::string graphFilePath);
	void Read();
	void ReadGraph();
	void ReadGraphGR();
	void SplitReadGraph(int);
	int max_nnz();
	void ReadGraphCSC();

	void Write2GR(std::string);
	void Write2CSCGR(std::string);
	void ReadCSC();
	void ReadGraphCSCGR();
};

Graph::Graph(std::string graphFilePath)
{
	this->graphFilePath = graphFilePath;
	this->weighted = true;
	hasZeroID = false;
	this->max_outdegree = 0;
}
int Graph::max_nnz()
{
	int max = 0;
	for (size_t i = 0; i < s_n; i++)
	{
		if (s_nnz[i] > max)
		{
			max = s_nnz[i];
		}
	}
	return max;
}
void Graph::Read()
{
	std::string s = graphFilePath.substr(graphFilePath.size() - 3, 3);
	if (s == ".gr")
	{
		std::ifstream fin(graphFilePath);
		if (!fin)
		{
			std::string filepath = graphFilePath;
			graphFilePath = graphFilePath.substr(0, graphFilePath.size() - 3) + ".tsv";
			ReadGraph();
			Write2GR(filepath);
		}
		else
			ReadGraphGR();
	}
	else
	{
		ReadGraph();
	}
}

void Graph::ReadCSC()
{
	std::string s = graphFilePath.substr(graphFilePath.size() - 3, 3);
	if (s == ".gr")
	{
		std::string filepath = graphFilePath.substr(0, graphFilePath.size() - 3) + "-CSC.gr";
		std::ifstream fin(filepath);
		if (!fin)
		{
			graphFilePath = graphFilePath = graphFilePath.substr(0, graphFilePath.size() - 3) + ".tsv";
			ReadGraphCSC();
			Write2CSCGR(filepath);
		}
		else
		{
			graphFilePath = filepath;
			ReadGraphCSCGR();
		}
	}
	else
	{
		ReadGraphCSC();
	}
}

void Graph::SplitReadGraph(int batch_size)
{
	// cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	std::ifstream infile;
	infile.open(graphFilePath);
	std::stringstream ss;
	int max = 0;
	int source = 0;
	int end = 0;
	double w8;
	std::string line;
	Edge newEdge;

	std::map<int, std::vector<Edge>> batch_graph;
	std::map<int, std::vector<double>> weights;

	// std::vector<int> line1;
	// std::vector<std::vector<int>> lines;
	int64_t edgeCounter = 0;
	while (getline(infile, line))
	{
		if (line[0] < '0' || line[0] > '9')
			continue;

		ss.str("");
		ss.clear();
		ss << line;

		ss >> source;
		ss >> end;
		ss >> w8;

		newEdge.source = (source - 1) % batch_size;
		newEdge.end = (end - 1);

		int batch_index = (source - 1) / batch_size;
		if (batch_graph.find(batch_index) != batch_graph.end())
		{
			batch_graph[batch_index].push_back(newEdge);
			weights[batch_index].push_back(w8);
		}
		else
		{
			batch_graph[batch_index] = std::vector<Edge>();
			batch_graph[batch_index].push_back(newEdge);
			weights[batch_index] = std::vector<double>();
			weights[batch_index].push_back(w8);
		}

		if (max < source)
			max = source;

		edgeCounter++;
	}
	num_edges = edgeCounter;
	num_nodes = max;
	num_nodes++;

	int batch_num = batch_graph.size();
	s_n = batch_num;
	s_xadj = new int *[batch_num];
	s_adjncy = new int *[batch_num];
	s_adjwgt = new double *[batch_num];
	s_outDegree = new int *[batch_num];
	s_nv = new int[batch_num]();
	s_nnz = new int[batch_num]();
	int i = 0;

	std::map<int, std::vector<Edge>>::iterator iter;
	iter = batch_graph.begin();
	while (iter != batch_graph.end())
	{
		int edge_size = iter->second.size();
		int node_size = 0;
		if (i < batch_num - 1)
		{
			node_size = batch_size;
		}
		else
		{
			node_size = num_nodes - batch_size * i;
		}
		s_nv[i] = node_size;
		s_nnz[i] = edge_size;
		s_outDegree[i] = new int[batch_size]();
		s_xadj[i] = new int[batch_size + 1];
		s_adjncy[i] = new int[edge_size]();
		s_adjwgt[i] = new double[edge_size]();
		for (int j = 0; j < edge_size; j++)
		{
			s_outDegree[i][iter->second[j].source]++;
			s_adjwgt[i][j] = weights[iter->first][j];
		}
		int counter = 0;
		for (int j = 0; j < batch_size; j++)
		{
			s_xadj[i][j] = counter;
		}
		for (int j = 0; j < node_size; j++)
		{
			s_xadj[i][j] = counter;
			counter = counter + s_outDegree[i][j];
			if (max_outdegree < s_outDegree[i][j])
			{
				max_outdegree = s_outDegree[i][j];
			}
		}
		for (size_t j = node_size; j < batch_size; j++)
		{
			s_xadj[i][j] = counter;
		}

		int *outDegreeCounter = new int[node_size]();
		for (int j = 0; j < edge_size; j++)
		{
			source = iter->second[j].source;
			end = iter->second[j].end;

			int location = s_xadj[i][source] + outDegreeCounter[source];
			s_adjncy[i][location] = end;
			outDegreeCounter[source]++;
		}
		i++;
		iter++;
	}
	infile.close();
}

void Graph::ReadGraph()
{
	// cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	std::ifstream infile;
	infile.open(graphFilePath);
	std::stringstream ss;
	int max = 0;
	int source;
	int end;
	double w8;
	// int i = 0;
	std::string line;
	Edge newEdge;
	// int h1, h2, h3;
	// std::vector<int> line1;
	// std::vector<std::vector<int>> lines;
	int64_t edgeCounter = 0;
	while (getline(infile, line))
	{
		if (line[0] < '0' || line[0] > '9')
			continue;

		ss.str("");
		ss.clear();
		ss << line;

		ss >> newEdge.source;
		ss >> newEdge.end;

		edges.push_back(newEdge);

		if (max < newEdge.source)
			max = newEdge.source;
		if (max < newEdge.end)
			max = newEdge.end;
		if (newEdge.source == 0)
			hasZeroID = true;
		if (newEdge.end == 0)
			hasZeroID = true;

		if (weighted)
		{
			if (ss >> w8)
				weights.push_back(w8);
			else
				weights.push_back(1);
		}

		edgeCounter++;
	}
	num_edges = edgeCounter;
	num_nodes = max;
	if (hasZeroID)
		num_nodes++;

	outDegree = new int[num_nodes]();

	for (int i = 0; i < num_edges; i++)
	{
		outDegree[edges[i].source - 1]++;
		// inDegree[graph.edges[i].end]++;
	}
	xadj = new int[num_nodes + 1];
	adjncy = new int[num_edges];
	adjwgt = new double[num_edges];
	long long counter = 0;
	for (int i = 0; i < num_nodes; i++)
	{
		xadj[i] = counter;
		counter = counter + outDegree[i];
		if (max_outdegree < outDegree[i])
		{
			max_outdegree = outDegree[i];
		}
	}
	int *outDegreeCounter = new int[num_nodes]();
	xadj[num_nodes] = counter;

	for (int i = 0; i < num_edges; i++)
	{
		source = edges[i].source - 1;
		end = edges[i].end - 1;
		// w8 =weights[i];

		int location = xadj[source] + outDegreeCounter[source];

		adjncy[location] = end;
		adjwgt[location] = weights[i];
		outDegreeCounter[source]++;
	}
	s_n = 1;
	infile.close();
}

void Graph::ReadGraphCSC()
{
	// cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	std::ifstream infile;
	infile.open(graphFilePath);
	std::stringstream ss;
	int max = 0;
	int source;
	int end;
	double w8;
	// int i = 0;
	std::string line;
	Edge newEdge;
	// int h1, h2, h3;
	// std::vector<int> line1;
	// std::vector<std::vector<int>> lines;
	int64_t edgeCounter = 0;
	while (getline(infile, line))
	{
		if (line[0] < '0' || line[0] > '9')
			continue;

		ss.str("");
		ss.clear();
		ss << line;
		ss >> newEdge.end;
		ss >> newEdge.source;

		edges.push_back(newEdge);

		if (max < newEdge.source)
			max = newEdge.source;
		if (max < newEdge.end)
			max = newEdge.end;
		if (newEdge.source == 0)
			hasZeroID = true;
		if (newEdge.end == 0)
			hasZeroID = true;
		if (weighted)
		{
			if (ss >> w8)
				weights.push_back(w8);
			else
				weights.push_back(1);
		}
		edgeCounter++;
	}
	num_edges = edgeCounter;
	num_nodes = max;
	if (hasZeroID)
		num_nodes++;
	outDegree = new int[num_nodes]();

	for (int i = 0; i < num_edges; i++)
	{
		outDegree[edges[i].source - 1]++;
		// inDegree[graph.edges[i].end]++;
	}
	csc_xadj = new int[num_nodes + 1];
	csc_adjncy = new int[num_edges];
	csc_adjwgt = new double[num_edges];
	long long counter = 0;
	for (int i = 0; i < num_nodes; i++)
	{
		csc_xadj[i] = counter;
		counter = counter + outDegree[i];
	}
	int *outDegreeCounter = new int[num_nodes]();
	csc_xadj[num_nodes] = counter;
	for (int i = 0; i < num_edges; i++)
	{
		source = edges[i].source - 1;
		end = edges[i].end - 1;
		// w8 =weights[i];

		int location = csc_xadj[source] + outDegreeCounter[source];

		csc_adjncy[location] = end;
		csc_adjwgt[location] = weights[i];
		outDegreeCounter[source]++;
	}
	infile.close();
}

void Graph::Write2CSCGR(std::string filename)
{
	FILE *fp;
	fp = fopen(filename.c_str(), "wb");
	fwrite(&num_nodes, sizeof(int), 1, fp);
	fwrite(&num_edges, sizeof(int64_t), 1, fp);
	fwrite(csc_xadj, sizeof(int), num_nodes + 1, fp);
	fwrite(csc_adjncy, sizeof(int), num_edges, fp);
	fwrite(csc_adjwgt, sizeof(double), num_edges, fp);
	fclose(fp);
}

void Graph::ReadGraphCSCGR()
{
	FILE *fp;
	fp = fopen(graphFilePath.c_str(), "r");
	fread(&num_nodes, sizeof(int), 1, fp);
	fread(&num_edges, sizeof(int64_t), 1, fp);
	csc_xadj = new int[num_nodes + 1];
	csc_adjncy = new int[num_edges];
	csc_adjwgt = new double[num_edges];
	fread(csc_xadj, sizeof(int), num_nodes + 1, fp);
	fread(csc_adjncy, sizeof(int), num_edges, fp);
	fread(csc_adjwgt, sizeof(double), num_edges, fp);
	fclose(fp);
}

void Graph::Write2GR(std::string filename)
{
	FILE *fp;
	fp = fopen(filename.c_str(), "wb");
	fwrite(&num_nodes, sizeof(int), 1, fp);
	fwrite(&num_edges, sizeof(int64_t), 1, fp);
	fwrite(xadj, sizeof(int), num_nodes + 1, fp);
	fwrite(adjncy, sizeof(int), num_edges, fp);
	fwrite(adjwgt, sizeof(double), num_edges, fp);
	fclose(fp);
}

void Graph::ReadGraphGR()
{
	FILE *fp;
	fp = fopen(graphFilePath.c_str(), "r");

	fread(&num_nodes, sizeof(int), 1, fp);
	fread(&num_edges, sizeof(int64_t), 1, fp);

	xadj = new int[num_nodes + 1];
	adjncy = new int[num_edges];
	adjwgt = new double[num_edges];

	fread(xadj, sizeof(int), num_nodes + 1, fp);
	fread(adjncy, sizeof(int), num_edges, fp);
	fread(adjwgt, sizeof(double), num_edges, fp);
	fclose(fp);
}

// FILE *gk_fopen(const char *fname, const char *mode, const char *msg)
// {
// 	FILE *fp;
// 	char errmsg[8192];

// 	fp = fopen(fname, mode);
// 	if (fp != NULL)
// 		return fp;

// 	sprintf(errmsg, "file: %s, mode: %s, [%s]", fname, mode, msg);
// 	perror(errmsg);
// 	printf("Failed on gk_fopen()\n");

// 	return NULL;
// }

// void Graph::ReadGraphGR()
// {
// 	// int *vsize;
// 	FILE *fpin;
// 	bool readew;

// 	fpin = gk_fopen(graphFilePath.data(), "r", "ReadGraphGR: Graph");

// 	size_t read;
// 	int64_t x[4];
// 	if (fread(x, sizeof(int64_t), 4, fpin) != 4)
// 	{
// 		printf("Unable to read header\n");
// 	}

// 	if (x[0] != 1) /* version */
// 		printf("Unknown file version\n");

// 	sizeEdgeTy = x[1];
// 	// int64_t sizeEdgeTy = le64toh(x[1]);
// 	int64_t numNode = x[2];
// 	int64_t numEdge = x[3];

// 	std::cout << graphFilePath + " has " << numNode << " nodes and " << numEdge << "  edges\n";
// 	xadj = new int[numNode + 1];
// 	adjncy = new int[numEdge];
// 	if (sizeEdgeTy)
// 	{
// 		adjwgt = new double[numEdge];
// 		weighted = true;
// 	}

// 	outDegree = new int[numNode];

// 	assert(xadj != NULL);
// 	assert(adjncy != NULL);
// 	//assert(vwgt != NULL);
// 	// assert(adjwgt != NULL);

// 	if (sizeof(int) == sizeof(int64_t))
// 	{
// 		read = fread(xadj + 1, sizeof(int), numNode, fpin); // This is little-endian data
// 		if (read < numNode)
// 			printf("Error: Partial read of node data\n");
// 		fprintf(stderr, "read %lu nodes\n", numNode);
// 	}
// 	else
// 	{
// 		for (int i = 0; i < numNode; i++)
// 		{
// 			int64_t rs;
// 			if (fread(&rs, sizeof(int64_t), 1, fpin) != 1)
// 			{
// 				printf("Error: Unable to read node data\n");
// 			}
// 			xadj[i + 1] = rs;
// 		}
// 	}

// 	// edges are 32-bit

// 	if (sizeof(int) == sizeof(int32_t))
// 	{
// 		read = fread(adjncy, sizeof(int), numEdge, fpin); // This is little-endian data
// 		if (read < numEdge)
// 			printf("Error: Partial read of edge destinations\n");

// 		// fprintf(stderr, "read %lu edges\n", numEdge);
// 	}
// 	else
// 	{
// 		assert(false && "Not implemented"); /* need to convert sizes when reading */
// 	}

// 	for (size_t i = 0; i < numNode; i++)
// 	{
// 		outDegree[i] = xadj[i + 1] - xadj[i];
// 	}

// 	if (sizeEdgeTy)
// 	{
// 		if (numEdge % 2)
// 			if (fseek(fpin, 4, SEEK_CUR) != 0) // skip
// 				printf("Error when seeking\n");

// 		if (sizeof(int) == sizeof(int32_t))
// 		{
// 			read = fread(adjwgt, sizeof(double), numEdge, fpin); // This is little-endian data
// 			readew = true;
// 			if (read < numEdge)
// 				printf("Error: Partial read of edge data\n");

// 			fprintf(stderr, "read data for %lu edges\n", numEdge);
// 		}
// 		else
// 		{
// 			assert(false && "Not implemented"); /* need to convert sizes when reading */
// 		}
// 	}
// 	num_nodes = numNode;
// 	num_edges = numEdge;
// }

#endif
