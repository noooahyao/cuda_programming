#ifndef _MATRIX_IO_CUH__
#define _MATRIX_IO_CUH__
#include "indexing_defs.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


/*
 * Header file for matrix reading, writing, and conversion
 * Read and write operations for matrices of 2, 3, and 4 dimensions
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * Matrices are read in and stored in row major order
 * However, cuSPARSE assumes matrices are stored in column major order
 * Use convert_to_column_major to convert the matrix
 */
 struct Matrix {
   double *vals;
   int dims[4];
   char is_column_first;
 };

/* ======================== Function declarations=========================== */
void read_matrix_vals(const char * filename, struct Matrix *mat,int is_column_first_flag);
void read_matrix_dims(const char * filename, struct Matrix *mat, int* product);
void print_matrix(struct Matrix *mat);
void convert_to_column_major(struct Matrix *matrix_row_major,
                            struct Matrix *matrix_col_major);
void convert_to_row_major(struct Matrix *matrix_row_major,
                            struct Matrix *matrix_col_major);
int isMatricesHaveSameDim(struct Matrix *matrix_row_major,
                            struct Matrix *matrix_col_major);
double calculateDistanceMatrix(struct Matrix *matrix1,struct Matrix *matrix2);
void initiliaze2dMatrix(struct Matrix *mat,int nRow,int nCol);
void destroyMatrix(struct Matrix *mat);
void transpose2dMatrix(struct Matrix *original,struct Matrix *transposed);

// void convert_dense_to_coo_4D(double *, double *, int *, int *, int *, int *, int [])
/* ========================================================================= */



void initiliaze2dMatrix(struct Matrix *mat,int nRow,int nCol){
  mat->dims[0] = mat->dims[1] = 0;
  mat->dims[2]=nRow;
  mat->dims[3]=nCol;

  mat->vals = (double *)calloc(nRow*nCol,sizeof(double));
}

void transpose2dMatrix(struct Matrix *original,struct Matrix *transposed){
  // Creates a new matrix, so destroy it.
  int temp;
  for (int i = 0; i < 4; i++)
  {
    transposed->dims[i] = original->dims[i];
  }
  int num_elems = transposed->dims[2] * transposed->dims[3]
                  * MAX(transposed->dims[0], 1)
                  * MAX(transposed->dims[1], 1);
  transposed->vals = (double *)calloc(num_elems, sizeof(double));
  transposed->is_column_first = original->is_column_first;
  if (original->is_column_first){//This is a code reuse. In the future implement convert_to_column major by using transpos maybe.
    convert_to_row_major(original,transposed);
  }
  else{
    convert_to_column_major(original,transposed);
  }
  transposed->is_column_first  ^= 1;
  temp = transposed->dims[2];
  transposed->dims[2] = transposed->dims[3];
  transposed->dims[3] = temp;
}
void destroyMatrix(struct Matrix *mat){
  mat->dims[0] = mat->dims[1]=mat->dims[2]=mat->dims[3] = 0;
  free(mat->vals);
}

double calculateDistanceMatrix(struct Matrix *matrix1,struct Matrix *matrix2){
  char flag;
  double result=0;
  long counter=0;
  int s, ch, i, j,iCol,iRow;
  if (isMatricesHaveSameDim(matrix1,matrix2)){
    if (matrix1->is_column_first == matrix2->is_column_first)
    {
      if(matrix1->is_column_first){
        flag=1 ;
        #ifdef DEBUG
        printf("Comparing two column-based matrices\n");
        #endif
      }
      else
      {
        flag=0 ;
        #ifdef DEBUG
        printf("Comparing two column-based matrices\n");
        #endif
      }
    }
    else{
      if(matrix1->is_column_first){
        flag=110;
        #ifdef DEBUG
        printf("Comparing mat1(col) with mat2(row)\n");
        #endif
      }
      else
      {
        flag=101;
        #ifdef DEBUG
        printf("Comparing mat1(row) with mat2(col)\n");
        #endif
      }
    }

    for (s = 0; s < MAX(matrix1->dims[0], 1); s++)
    {
      for (ch = 0; ch < MAX(matrix1->dims[1], 1); ch++)
      {
        for (i = 0; i < matrix1->dims[2]; i++)
        {
          for (j = 0; j < matrix1->dims[3]; j++)
          {
            if (flag==1){
              iCol = index4DCol(s, ch, i, j,
                matrix1->dims[1], matrix1->dims[2], matrix1->dims[3]);
              result += fabsf(matrix1->vals[iCol] - matrix2->vals[iCol]);
            }
            else if(flag==0){
              iRow = index4D(s, ch, i, j,
                matrix1->dims[1], matrix1->dims[2], matrix1->dims[3]);
              result += fabsf(matrix1->vals[iRow] - matrix2->vals[iRow]);
            }
            else if(flag==110){
              iCol = index4DCol(s, ch, i, j,
                matrix1->dims[1], matrix1->dims[2], matrix1->dims[3]);
              iRow = index4D(s, ch, i, j,
                matrix1->dims[1], matrix1->dims[2], matrix1->dims[3]);
              result += fabsf(matrix1->vals[iCol] - matrix2->vals[iRow]);
            }
            else if(flag==101){
              iCol = index4DCol(s, ch, i, j,
                matrix1->dims[1], matrix1->dims[2], matrix1->dims[3]);
              iRow = index4D(s, ch, i, j,
                matrix1->dims[1], matrix1->dims[2], matrix1->dims[3]);
              result += fabsf(matrix1->vals[iRow] - matrix2->vals[iCol]);
            }
            counter ++;
          }
        }
      }
    }
    #ifdef DEBUG
    printf("Total elements:%ld\n",counter);
    #endif
    return result/counter;
  }
  else{
    printf("[WARNING]: You just tried to compare to matrices with different sizes\n");
    return -1;
  }
}

int isMatricesHaveSameDim(struct Matrix *matrix_row_major,
                             struct Matrix *matrix_col_major)
 {
   for (int i =0;i<4;i++){
     if (matrix_row_major->dims[i] != matrix_col_major->dims[i]){
       return 0;
     }
   }
   return 1;
 }

void convert_to_column_major(struct Matrix *matrix_row_major,
                             struct Matrix *matrix_col_major)
{
  // Write planes
  int s, ch, i, j;
  // TODO check whether they have same dimensions.
  if (! isMatricesHaveSameDim(matrix_row_major,matrix_col_major)){
    printf("Matrix dimensions doesn't match\n");
    exit(1);
  }
  else if (matrix_row_major->is_column_first!=0){
    printf("First matrix should be row major ordered\n");
    exit(1);
  }

  //Otherwise lets change
  for (s = 0; s < MAX(matrix_row_major->dims[0], 1); s++)
  {
    for (ch = 0; ch < MAX(matrix_row_major->dims[1], 1); ch++)
    {
      for (i = 0; i < matrix_row_major->dims[2]; i++)
      {
        for (j = 0; j < matrix_row_major->dims[3]; j++)
        {
          matrix_col_major->vals[index4DCol(s, ch, i, j,
            matrix_row_major->dims[1], matrix_row_major->dims[2], matrix_row_major->dims[3])] =
            matrix_row_major->vals[index4D(s, ch, i, j,
            matrix_row_major->dims[1], matrix_row_major->dims[2], matrix_row_major->dims[3])];
        }
      }
    }
  }
  matrix_col_major->is_column_first=1;
}


void convert_to_row_major(struct Matrix *matrix_col_major,
                             struct Matrix *matrix_row_major)
{
  // Write planes
  int s, ch, i, j;
  if (! isMatricesHaveSameDim(matrix_row_major,matrix_col_major)){
    printf("Matrix dimensions doesn't match\n");
    exit(1);
  }
  else if (matrix_col_major->is_column_first!=1){
    printf("Col major matrix should be col major ordered\n");
    exit(1);
  }

  //Otherwise lets change
  for (s = 0; s < MAX(matrix_row_major->dims[0], 1); s++)
  {
    for (ch = 0; ch < MAX(matrix_row_major->dims[1], 1); ch++)
    {
      for (i = 0; i < matrix_row_major->dims[2]; i++)
      {
        for (j = 0; j < matrix_row_major->dims[3]; j++)
        {
          matrix_row_major->vals[index4D(s, ch, i, j,
            matrix_col_major->dims[1], matrix_col_major->dims[2], matrix_col_major->dims[3])] =
            matrix_col_major->vals[index4DCol(s, ch, i, j,
            matrix_col_major->dims[1], matrix_col_major->dims[2], matrix_col_major->dims[3])];
        }
      }
    }
  }
  matrix_row_major->is_column_first=0;
}


void read_matrix_dims(const char * filename, struct Matrix *mat ,int* product)
{
  // Return the multiplication of the dimensions, a.k. number of elements
  FILE *fp = fopen(filename, "r");
  size_t len = 0;
  char *line = NULL;
  getline(&line, &len, fp);
  int dim = atoi(&line[6]);

  #ifdef DEBUG
    printf("DIM: %d\n", dim);
    printf("%s", line);
  #endif

  int i;
  //initiliaze dimensions to 0
  for (i = 0; i < 4; i ++)
  {
    mat->dims[i]=0;
  }

  // lets read the dims
  *product = 1;
  int offset = 4 - dim;
  for (i = 0; i < dim; i ++)
  {
    getline(&line, &len, fp);
    mat->dims[i + offset] = atoi(&line[5]);
    *product *= mat->dims[i + offset];

    #ifdef DEBUG
        printf("%d: %s", i, line);
    #endif
  }
  #ifdef DEBUG
    printf("Matrix dimensions: [%d, %d, %d, %d]\n",
                                mat->dims[0],
                                mat->dims[1],
                                mat->dims[2],
                                mat->dims[3]);
  #endif

  free(line);
  fclose(fp);
}


void read_matrix_vals(const char * filename, struct Matrix *mat,int is_column_first_flag)
{
  FILE *fp = fopen(filename, "r");
  size_t len = 0;
  char *line = NULL;
    // Discard matrix descriptor rows and gather matrix stats
  getline(&line, &len, fp);
  int k;
  for (k = 0; k < 4; k++)
  {
    if (mat->dims[k] != 0)
    {
      // printf("Dim[%d]: %d\n", k,mat->dims[k]);
      getline(&line, &len, fp);
    }
  }
  // Read planes
  int s, ch, i, j;
  for (s = 0; s < MAX(mat->dims[0], 1); s++)
  {
    for (ch = 0; ch < MAX(mat->dims[1], 1); ch++)
    {
      // Read Matrix breaker line
      getline(&line, &len, fp);
       // printf("Line: %s\n", line);
      for (i = 0; i < mat->dims[2]; i++)
      {
        // Read row
        getline(&line, &len, fp);
        // printf("Row length: %d Line: %s\n", len, line);
        char * token;
        const char * delim = ",";
        token = strtok(line, delim);
        for (j = 0; j < mat->dims[3]; j++)
        {
          // printf("[%d, %d, %d, %d]\n", s, ch, i, j);
          double tok_f = atof(token);
          if (is_column_first_flag){
            mat->vals[index4DCol(s, ch, i, j, mat->dims[1], mat->dims[2], mat->dims[3])] = tok_f;
          }
          else{
            mat->vals[index4D(s, ch, i, j, mat->dims[1], mat->dims[2], mat->dims[3])] = tok_f;
          }
          token = strtok(NULL, delim);
        }
      }
    }
  }
  mat->is_column_first = is_column_first_flag;
  free(line);
  fclose(fp);
}


void print_matrix(struct Matrix *mat)
{
  printf("Matrix dimensions: [%d, %d, %d, %d]\n",
                              mat->dims[0],
                              mat->dims[1],
                              mat->dims[2],
                              mat->dims[3]);
  // Write planes
  if (mat->is_column_first)
  {
    printf("Column major ordering\n");
  }
  else
  {
    printf("Row major ordering\n");
  }
  int s, ch, i, j;
  for (s = 0; s < MAX(mat->dims[0], 1); s++)
  {
    for (ch = 0; ch < MAX(mat->dims[1], 1); ch++)
    {
      printf("(Sample, Channel) = (%d, %d)\n", s, ch);

        for (i = 0; i < mat->dims[2]; i++)
        {
          for (j = 0; j < mat->dims[3]; j++)
          {
            if (mat->is_column_first){
              printf("%.7f ", mat->vals[index4DCol(s, ch, i, j,
                mat->dims[1], mat->dims[2], mat->dims[3])]);
            }
            else{
              printf("%.7f ", mat->vals[index4D(s, ch, i, j,
                mat->dims[1], mat->dims[2], mat->dims[3])]);
            }
          }
          printf("\n");
        }
        printf("\n");
    }
  }
}

#endif