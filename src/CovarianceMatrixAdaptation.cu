/* Functions  For Covariance Matrix Adaptation */

#include "../include/CovarianceMatrixAdaptation.cuh"


__global__ void cpy_Previous_CovarianceMatrix(float *OutCov, float *inCov)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    OutCov[id] = inCov[id];
    __syncthreads( );
}

__global__ void make_Covariance_Matrix(float *dCov,float *bCov, float *mean, InputVector *d_Datas, int *indecies, int sz_smpl)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    float elements;
    // int denominator = 1 / (sz_smpl - 1);
    float f_denominator = 0.0f;
    // /分母（重みの和）を計算　←　重み付き標本共分散行列を使用する際 
    for(int num = 0; num < sz_smpl; num++)
    {
        if(isnan(d_Datas[indecies[num]].W))
        {
            f_denominator += 0.0f;
        }else{
            f_denominator += d_Datas[indecies[num]].W; 
        }
    }
    float weight_now;
    for(int i = 0; i < sz_smpl; i++)
    {
        weight_now = d_Datas[indecies[i]].W / f_denominator;
        // w  * (OP(y)-C^T) を愚直に計算した。
        elements += weight_now * ( (d_Datas[indecies[i]].Input[threadIdx.x] - mean[threadIdx.x]) * (d_Datas[indecies[i]].Input[threadIdx.x] - mean[threadIdx.x]) - bCov[id]);
    }

    dCov[id] = elements;
    __syncthreads( );
}

__global__ void result_CMA_rank_mu_Update(float *nGradCov, float *initCov, float learningRate)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    float temp;
    temp =  initCov[id] + learningRate * nGradCov[id];
    __syncthreads();
    nGradCov[id] = temp;
    __syncthreads();
}