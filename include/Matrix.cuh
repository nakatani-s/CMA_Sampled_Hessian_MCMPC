#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"

void printMatrix(int m, int n, float*A, int lda, const char* name);

unsigned int countBlocks(unsigned int a, unsigned int b);


__global__ void setup_Identity_Matrix(float *IdMat);
__global__ void setup_Identity_Matrix_overMaxThread(float *IdMat, int Ydimention);
// __global__ void make_Covariance_Matrix(float *dCov,float *bCov, float *mean, InputVector d_Datas, int *indecies, int sz_smpl);
// __global__ void make_Eigen_Diagonal_Matrix(float *DiagMat, float *VecEig);
__global__ void make_SqrtEigen_Diagonal_Matrix(float *DiagMat, float *VecEig);
__global__ void make_InverseEigen_Diagonal_Matrix(float *DiagMat, float *VecEig);
__global__ void copy_device_Matrix(float *Out, float *In);
__global__ void pwr_matrix_answerLater(float *A, float *B);
__global__ void copy_inputSequences(InputVector *outInput, float *temp);
__global__ void multiply_matrix(float *OutMatrix, float voc, float *InMatrix);