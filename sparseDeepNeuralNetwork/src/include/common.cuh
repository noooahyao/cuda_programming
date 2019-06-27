#ifndef COMMON_CUH
#define COMMON_CUH

#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <string>
#include <stdio.h>
#include <locale>
#include <sstream>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <stdexcept>
#include <sstream> 
#include <sys/types.h>  
#include <sys/stat.h> 
#include <iterator>
#include <map>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda.h>
#include "safe_call_defs.h"
#include "timer.cuh"
#include "graph.hpp"
#include "sparse_conversion.cuh"

using std::string; 
using std::cout;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::flush;
using std::to_string;
using std::stringstream;
using std::vector;
using uint= unsigned int ;

#endif