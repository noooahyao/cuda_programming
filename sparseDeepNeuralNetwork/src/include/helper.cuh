#ifndef __HELPER_CUH
#define __HELPER_CUH

#include <cuda.h>

// HandleError
static void 
HandleError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", \
    cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
  }
}

#define H_ERR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif