/* 
    MCMPC.cuh
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


// MCMPCによる推定入力列を計算する関数
void weighted_mean(InputVector *In, int numElite, float *Out);

// 制御周期の以降に伴う入力列のシフトのための関数
void shift_Input_vec( float *inputVector); 

// InputVector型の変数を初期化
__global__ void init_Input_vector(InputVector *d_I, float init_val);


__global__ void callback_elite_sample(InputVector *devOut, InputVector *devIn, int *elite_indices);
__global__ void setup_kernel(curandState *state,int seed);


__global__ void MCMPC_Simple_NonLinear_Example(float *state, curandState *randomSeed , float *mean, InputVector *d_data, float var, float *d_param, float *d_constraints, float *d_matrix, float *cost_vec);

__global__ void MCMPC_Crat_and_SinglePole(float *state, curandState *randomSeed, float *mean, InputVector *d_data, float var, float *d_param, float *d_constraints, float  *d_matrix, float *cost_vec);

__global__ void CMAMCMPC_Cart_and_SinglePole(float *state, curandState *r_seed, float *covariance, float *mean, InputVector *d_data, float variance, float *d_param, float *d_constraints, float *d_matrix, float *cost_vec);