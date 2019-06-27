
// /*
//  * Matrix multiplication experiments.
//  */
// #include "matrix_io.cuh"
// #include "mm.cuh"
// #include <stdlib.h>
// #include <stdio.h>
// #include <cublas.h>
// #include <cusparse.h>
// #include <unistd.h>
// #include <ctype.h>

// // #define DEBUG

// int main(int argc, char *argv[])
// {
//     struct Matrix matrix1, matrix2, matrixResCPU, matrixResGPU;
//     int n_elements;
//     const char *filename1;
//     const char *filename2;
//     double time_taken;
//     clock_t start, end;
//     float diff;

//     int time_flag = 0;
//     int repetation_n = 0;
//     int correctness_check_flag = 0;
//     char *alg_type_flag = NULL;

//     filename1 = argv[optind];
//     filename2 = argv[optind + 1];

//     cudaFree(0);
//     read_matrix_dims(filename1, &matrix1, &n_elements);
//     matrix1.vals = (float *)calloc(n_elements, sizeof(float));
//     read_matrix_vals(filename1, &matrix1, 1);

//     read_matrix_dims(filename2, &matrix2, &n_elements);
//     matrix2.vals = (float *)calloc(n_elements, sizeof(float));
//     read_matrix_vals(filename2, &matrix2, 1);

//     int m = matrix1.dims[2];
//     int k = matrix1.dims[3];
//     int n = matrix2.dims[3];

//     if (correctness_check_flag)
//     {
//         //cpu_mm
//         initiliaze2dMatrix(&matrixResCPU, m, n);
//         start = clock();
//         cpu_mm(&matrix1, &matrix2, &matrixResCPU, m, n, k);
//         end = clock();
//         time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
//         printf("Time taken for the cpu_mm is %lf\n", time_taken);
//     }

//     if (strcmp(alg_type_flag, "denseblas") == 0)
//     {
//         ///gpu_mm_dense
//         // Create a handle for CUBLAS
//         initiliaze2dMatrix(&matrixResGPU, m, n);
//         cublasHandle_t handleBLAS;
//         cublasCreate(&handleBLAS);

//         start = clock();
//         gpu_mm_dense(&matrix1, &matrix2, &matrixResGPU, m, n, k, handleBLAS, time_flag, repetation_n);
//         end = clock();
//         time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
//         printf("Time taken for the gpu_mm_dense is %lf\n", time_taken);
// #ifdef DEBUG
//         printf("CuBLAS result:\n");
//         print_matrix(&matrixResGPU);
// #endif
//         if (correctness_check_flag)
//         {
//             diff = calculateDistanceMatrix(&matrixResCPU, &matrixResGPU);
//             if (diff > 1e-7)
//             {
//                 printf("Diff is %.8f\n", diff);
//                 printf("There might be a problem\n");
//             }
//             else
//             {
//                 printf("Diff is less then 1e-7\n", diff);
//             }
//         }
//         destroyMatrix(&matrixResGPU);
//         cublasDestroy(handleBLAS);
//     }
//     else if (strcmp(alg_type_flag, "cusparse") == 0)
//     {
//         ///gpu_mm_sparse
//         // Initialize cusparse library
//         initiliaze2dMatrix(&matrixResGPU, m, n);
//         cusparseHandle_t handleSparse;
//         cusparseCreate(&handleSparse);

//         start = clock();
//         gpu_mm_sparse(&matrix1, &matrix2, &matrixResGPU, m, n, k, handleSparse, time_flag, repetation_n);
//         end = clock();
//         time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
//         printf("Time taken for gpu_mm_sparse is %lf\n", time_taken);
// #ifdef DEBUG
//         printf("cuSparse result:\n");
//         print_matrix(&matrixResGPU);
// #endif
//         if (correctness_check_flag)
//         {
//             diff = calculateDistanceMatrix(&matrixResCPU, &matrixResGPU);
//             if (diff > 1e-7)
//             {
//                 printf("Diff is %.8f\n", diff);
//                 printf("There might be a problem\n");
//             }
//             else
//             {
//                 printf("Diff is less then 1e-7\n", diff);
//             }
//         }
//         cusparseDestroy(handleSparse);
//         destroyMatrix(&matrixResGPU);
//     }
//     else if (strcmp(alg_type_flag, "sparseimp") == 0)
//     {
//         //gpu_mm_sparse_ProjectImp
//         initiliaze2dMatrix(&matrixResGPU, m, n);
//         cusparseHandle_t handleSparse;
//         cusparseCreate(&handleSparse);
//         start = clock();
//         gpu_mm_sparse_ProjectImp(&matrix1, &matrix2, &matrixResGPU, m, n, k, handleSparse, time_flag, repetation_n);
//         end = clock();
//         time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
//         printf("Time taken for gpu_mm_sparse_ProjectImp is %lf\n", time_taken);
// #ifdef DEBUG
//         printf("cuSparse result:\n");
//         print_matrix(&matrixResGPU);
// #endif
//         if (correctness_check_flag)
//         {
//             diff = calculateDistanceMatrix(&matrixResCPU, &matrixResGPU);
//             if (diff > 1e-7)
//             {
//                 printf("Diff is %.8f\n", diff);
//                 printf("There might be a problem\n");
//             }
//             else
//             {
//                 printf("Diff is less then 1e-7\n", diff);
//             }
//         }
//         cusparseDestroy(handleSparse);
//         destroyMatrix(&matrixResGPU);
//     }
//     else
//     {
//         printf("Use denseblas/cusparse/sparseimp with -a flag.\n");
//     }

//     destroyMatrix(&matrix1);
//     destroyMatrix(&matrix2);
//     if (correctness_check_flag)
//     {
//         destroyMatrix(&matrixResCPU);
//     }
// }