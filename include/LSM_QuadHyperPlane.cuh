/*
    LSM_QuadHyperPlane.cuh
*/

#include<cuda.h>
#include<curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"


unsigned int count_QHP_Parameters(int hor);
__global__ void LSM_QHP_make_tensor_vector(QuadHyperPlane *output, InputVector *input, int *indices);
__global__ void LSM_QHP_make_regular_matrix_over_ThreadPerBlockLimit(float *outRmatrix, QuadHyperPlane *elements, int sumSet, int Ydimention);
__global__ void LSM_QHP_make_regular_matrix(float *outRmatrix, QuadHyperPlane *elements, int sumSet);
__global__ void LSM_QHP_make_regular_vector(float *outRvector, QuadHyperPlane *elements, int sumSet );
__global__ void LSM_QHP_get_reslt_all_elements(float *outElements, float *inElements);
__global__ void LSM_QHP_get_Hessian_Result(float *outElements, float *inElements);
__global__ void LSM_QHP_make_Hvector(float *Output, InputVector *datas, float *Hess);
__global__ void LSM_QHP_make_transGmatrix(float *Output, InputVector *datas);

__global__ void LSM_QHP_make_bVector(float *OutVector, float *Elemets, int indecies);
__global__ void LSM_QHP_make_symmetric(float *Out, float *In);
__global__ void LSM_QHP_transpose(float *Out, float *In);

__global__ void LSM_Hessian_To_Positive_Symmetric(float *Hess);