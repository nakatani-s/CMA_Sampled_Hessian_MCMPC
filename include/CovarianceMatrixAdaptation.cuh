#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"

// 前更新時の分散共分散行列をコピーする関数
__global__ void cpy_Previous_CovarianceMatrix(float *OutCov, float *inCov);
// サンプル点情報を使用して共分散行列を計算・生成する関数
__global__ void make_Covariance_Matrix(float *dCov,float *bCov, float *mean, InputVector *d_Datas, int *indecies, int sz_smpl);
// 共分散行列Cの自然勾配にアップデート(自然勾配、初期値、学習係数（ニュートン法などを参考に）)
__global__ void result_CMA_rank_mu_Update(float *nGradCov, float *initCov, float learningRate);