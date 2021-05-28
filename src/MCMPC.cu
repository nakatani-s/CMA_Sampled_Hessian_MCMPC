/* MCMPC.cu */
#include<stdio.h>
#include "../include/MCMPC.cuh"

void weighted_mean(InputVector *In, int numElite, float *Out)
{
    float totalWeight = 0.0f;
    float temp[HORIZON] = { };
    for(int i = 0; i < numElite; i++){
        if(isnan(In[i].W))
        {
            totalWeight += 0.0f;
        }else{
            totalWeight += In[i].W;
        }
    }
    for(int i = 0; i < HORIZON; i++){
        for(int k = 0; k < numElite; k++){
            if(isnan(In[k].W))
            {
                temp[i] += 0.0f;
            }else{
                temp[i] += (In[k].W * In[k].Input[i]) / totalWeight;
            }
        }
        if(isnan(temp[i]))
        {
            Out[i] = 0.0f;
        }else{
            Out[i] = temp[i];
        }
    }
}

void shift_Input_vec( float *inputVector)
{
    float temp[HORIZON]= { };
    for(int i = 0; i < HORIZON - 1; i++){
        temp[i] = inputVector[i+1];
    }
    temp[HORIZON - 1] = inputVector[HORIZON - 1];
    for(int i = 0; i < HORIZON; i++){
        inputVector[i] = temp[i];
    }
}

__global__ void init_Input_vector(InputVector *d_I, float init_val)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    for(int tm = 0; tm < HORIZON; tm++)
    {
        d_I[id].Input[tm] = init_val;
    }
}

__global__ void setup_kernel(curandState *state,int seed) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, id, 0, &state[id]);
}

__global__ void callback_elite_sample(InputVector *devOut, InputVector *devIn, int *elite_indices)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    devOut[id].W =  devIn[elite_indices[id]].W;
    devOut[id].L =  devIn[elite_indices[id]].L;
    for(int i = 0; i < HORIZON; i++){
        devOut[id].Input[i] = devIn[elite_indices[id]].Input[i];
        // devOut[id].dy[i] = devIn[elite_indices[id]].dy[i];
    }
}


__device__ float gen_u(unsigned int id, curandState *state, float ave, float vr) {
    float u;
    curandState localState = state[id];
    u = curand_normal(&localState) * vr + ave;
    return u;
}

__device__ float inputGenerator(int t, float mean, float var, float *Cov, float *z)
{
    int index;
    index = t * HORIZON;
    float ans, temp;
    temp = 0.0f;
    for(int k = 0; k < HORIZON; k++)
    {
        temp += Cov[ index + k ] * z[k];
    }

    ans = mean + var * temp;
    return ans;
}

//MCMPC for Simple Nonlinear System
__global__ void MCMPC_Simple_NonLinear_Example(float *state, curandState *randomSeed , float *mean, InputVector *d_data, float var, float *d_param, float *d_constraints,
    float *d_matrix, float *cost_vec){

    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq;
    seq = id;
    float qx = 0.0f;
    float total_cost = 0.0f;
    float u[HORIZON] = { };
    float stateInThisThreads[DIM_OF_STATES] = { };
    float dstateInThisThreads[DIM_OF_STATES] = { };

    //copy statevector for calculate forward simulation result in each thread. 
    for(int i = 0; i < DIM_OF_STATES; i++){
        stateInThisThreads[i] = state[i];
    }
    // do simulation in each thread
    for(int step = 0; step < HORIZON; step++)
    {
        u[step] = gen_u(seq, randomSeed, mean[step], var);
        seq += NUM_OF_SAMPLES;
        if(isnan(u[step])){
            u[step] = d_data[0].Input[step];
        }
        //printf("id==%d  u==%f\n", id, u[step]);
        calc_nonLinear_example(stateInThisThreads, u[step], d_param, dstateInThisThreads);
        stateInThisThreads[0] = stateInThisThreads[0] + (interval * dstateInThisThreads[0]);
        stateInThisThreads[1] = stateInThisThreads[1] + (interval * dstateInThisThreads[1]);

        qx = stateInThisThreads[0] * stateInThisThreads[0] * d_matrix[0] + stateInThisThreads[1] * stateInThisThreads[1] * d_matrix[1] + d_matrix[2] * u[step] * u[step];
        total_cost += qx;
        qx = 0.0f;
    }
    //printf("id==%d  L==%f   %f %f %f %f\n", id, total_cost, u[0], u[1], u[2], u[3]);

    float KL_COST, S, lambda;
    lambda = 200;/*HORIZON * DIM_OF_STATES;*/
    S = total_cost / lambda;
    KL_COST = exp(-S);
    //printf("id==%d  L==%f  W == %f  %f %f %f %f\n", id, total_cost, KL_COST, u[0], u[1], u[2], u[3]);
    __syncthreads();
    d_data[id].W = KL_COST;
    d_data[id].L = total_cost;
    cost_vec[id] = total_cost;
    for(int i = 0; i < HORIZON; i++){
        d_data[id].Input[i] = u[i];
    }
    __syncthreads();
}


__global__ void MCMPC_Crat_and_SinglePole(float *state, curandState *randomSeed, float *mean, InputVector *d_data, float var, float *d_param, float *d_constraints, float  *d_matrix, float *cost_vec)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq;
    seq = id;
    float qx = 0.0f;
    float total_cost = 0.0f;
    float u[HORIZON] = { };
    float stateInThisThreads[DIM_OF_STATES] = { };
    float dstateInThisThreads[DIM_OF_STATES] = { };

    for(int i = 0; i < DIM_OF_STATES; i++){
        stateInThisThreads[i] = state[i];
    }

    for(int t = 0; t < HORIZON; t++){
        
        if(isnan(mean[t])){
            //u[t] = d_data[0].Input[t];
            if(t < HORIZON -1){
                u[t] = gen_u(seq, randomSeed, d_data[0].Input[t+1], var);
            seq += NUM_OF_SAMPLES;
            }else{
                u[t] = gen_u(seq, randomSeed, d_data[0].Input[HORIZON - 1], var);
                seq += NUM_OF_SAMPLES;
            }
        }else{
            u[t] = gen_u(seq, randomSeed, mean[t], var);
            seq += NUM_OF_SAMPLES;
        }

        if(u[t] < d_constraints[0]){
            u[t] = d_constraints[0];
        }
        if(u[t] > d_constraints[1]){
            u[t] = d_constraints[1];
        }
        // まずは、オイラー積分（100Hz 40stepで倒立できるか）　→　0.4秒先まで予測
        // 問題が起きたら、0次ホールダーでやってみる、それでもダメならMPCの再設計
        /*dstateInThisThreads[0] = stateInThisThreads[2];
        dstateInThisThreads[1] = stateInThisThreads[3];
        dstateInThisThreads[2] = Cart_type_Pendulum_ddx(u[t], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param); //ddx
        dstateInThisThreads[3] = Cart_type_Pendulum_ddtheta(u[t], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param);
        stateInThisThreads[2] = stateInThisThreads[2] + (interval * dstateInThisThreads[2]);
        stateInThisThreads[3] = stateInThisThreads[3] + (interval * dstateInThisThreads[3]);
        stateInThisThreads[0] = stateInThisThreads[0] + (interval * dstateInThisThreads[0]);
        stateInThisThreads[1] = stateInThisThreads[1] + (interval * dstateInThisThreads[1]);*/
        for(int sec = 0; sec < 2; sec++){
            dstateInThisThreads[0] = stateInThisThreads[2];
            dstateInThisThreads[1] = stateInThisThreads[3];
            dstateInThisThreads[2] = Cart_type_Pendulum_ddx(u[t], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param); //ddx
            dstateInThisThreads[3] = Cart_type_Pendulum_ddtheta(u[t], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param);
            stateInThisThreads[2] = stateInThisThreads[2] + (interval * dstateInThisThreads[2]);
            stateInThisThreads[3] = stateInThisThreads[3] + (interval * dstateInThisThreads[3]);
            stateInThisThreads[0] = stateInThisThreads[0] + (interval * dstateInThisThreads[0]);
            stateInThisThreads[1] = stateInThisThreads[1] + (interval * dstateInThisThreads[1]);
        }

        while(stateInThisThreads[1] > M_PI)
            stateInThisThreads[1] -= (2 * M_PI);
        while(stateInThisThreads[1] < -M_PI)
            stateInThisThreads[1] += (2 * M_PI);

        // upper side: MATLAB　で使用している評価関数を参考    
        /* qx = stateInThisThreads[0] * stateInThisThreads[0] * d_matrix[0] + stateInThisThreads[1] * stateInThisThreads[1] * d_matrix[1]
            + u[t] * u[t] * d_matrix[3]; */
        qx = stateInThisThreads[0] * stateInThisThreads[0] * d_matrix[0] + stateInThisThreads[1] * stateInThisThreads[1] * d_matrix[1]
            + stateInThisThreads[2] * stateInThisThreads[2] * d_matrix[2] + stateInThisThreads[3] * stateInThisThreads[3] * d_matrix[3]
            + u[t] * u[t] * d_matrix[4];
        
        // constraints described by Barrier Function Method
        if(stateInThisThreads[0] <= 0){
            qx += 1 / (powf(stateInThisThreads[0] - d_constraints[2],2) * invBarrier);
            if(stateInThisThreads[0] < d_constraints[2]){
                qx += 1000000;
            }
        }else{
            qx += 1 / (powf(d_constraints[3] - stateInThisThreads[0],2) * invBarrier);
            if(stateInThisThreads[0] > d_constraints[3]){
                qx += 1000000;
            }
        }

        total_cost += qx;

        qx = 0.0f;
    }

    if(isnan(total_cost))
    {
        total_cost = 1000000 * 5;
    }
    float KL_COST, S, lambda, HM_COST, HM;
    //int NomarizationCost = sizeOfParaboloidElements + addTermForLSM; //LSMでparaboloidをフィッティングする際に行列の
    lambda = 2 * HORIZON;
    HM = total_cost / (0.75*HORIZON); //0.75
    S = total_cost / lambda;
    KL_COST = exp(-S);
    HM_COST = exp(-HM);
    __syncthreads();
    d_data[id].WHM = HM_COST;
    d_data[id].W = KL_COST;
    d_data[id].L = total_cost / sizeOfParaboloidElements;
    //d_data[id].L = total_cost;
    cost_vec[id] = total_cost;
    for(int index = 0; index < HORIZON; index++){
        d_data[id].Input[index] = u[index];
    }
    __syncthreads();

}

__global__ void CMAMCMPC_Cart_and_SinglePole(float *state, curandState *r_seed, float *covariance, float *mean, InputVector *d_data, float variance, float *d_param, float *d_constraints, float *d_matrix, float *cost_vec)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq; //種のインデックス（これを更新しないと全く同じ乱数が生成される）
    float qx = 0.0f; //stage_cost
    float total_cost = 0.0f; //total_cost
    float u[HORIZON] = { }; //シミュレーションに使用する入力
    float z[HORIZON] = { }; //↑の入力から前回の推定値分平行移動した入力

    float stateInThisThreads[DIM_OF_STATES] = { };
    float dstateInThisThreads[DIM_OF_STATES] = { };

    for(int i = 0; i < DIM_OF_STATES; i++){
        stateInThisThreads[i] = state[i];
    }

    for(int t_u = 0; t_u < HORIZON; t_u++)
    {
        z[t_u] = gen_u( id, r_seed, 0.0f, 1.0f);
        // seq += NUM_OF_SAMPLES;
        seq += HORIZON;
    }
    __syncthreads( );

    for(int t = 0; t < HORIZON; t++)
    {
        u[t] = inputGenerator(t, mean[t], variance, covariance, z);

        // 入力制約の実装
        if(u[t] < d_constraints[0]){
            u[t] = d_constraints[0];
        }
        if(u[t] > d_constraints[1]){
            u[t] = d_constraints[1];
        }

        for(int sec = 0; sec < 2; sec++){
            dstateInThisThreads[0] = stateInThisThreads[2];
            dstateInThisThreads[1] = stateInThisThreads[3];
            dstateInThisThreads[2] = Cart_type_Pendulum_ddx(u[t], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param); //ddx
            dstateInThisThreads[3] = Cart_type_Pendulum_ddtheta(u[t], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param);
            stateInThisThreads[2] = stateInThisThreads[2] + (interval * dstateInThisThreads[2]);
            stateInThisThreads[3] = stateInThisThreads[3] + (interval * dstateInThisThreads[3]);
            stateInThisThreads[0] = stateInThisThreads[0] + (interval * dstateInThisThreads[0]);
            stateInThisThreads[1] = stateInThisThreads[1] + (interval * dstateInThisThreads[1]);
        }

        while(stateInThisThreads[1] > M_PI)
            stateInThisThreads[1] -= (2 * M_PI);
        while(stateInThisThreads[1] < -M_PI)
            stateInThisThreads[1] += (2 * M_PI);

        // upper side: MATLAB　で使用している評価関数を参考    
        /* qx = stateInThisThreads[0] * stateInThisThreads[0] * d_matrix[0] + stateInThisThreads[1] * stateInThisThreads[1] * d_matrix[1]
            + u[t] * u[t] * d_matrix[3]; */
        qx = stateInThisThreads[0] * stateInThisThreads[0] * d_matrix[0] + stateInThisThreads[1] * stateInThisThreads[1] * d_matrix[1]
            + stateInThisThreads[2] * stateInThisThreads[2] * d_matrix[2] + stateInThisThreads[3] * stateInThisThreads[3] * d_matrix[3]
            + u[t] * u[t] * d_matrix[4];
        
        // constraints described by Barrier Function Method
        if(stateInThisThreads[0] <= 0){
            qx += 1 / (powf(stateInThisThreads[0] - d_constraints[2],2) * invBarrier);
            if(stateInThisThreads[0] < d_constraints[2]){
                qx += 1000000;
            }
        }else{
            qx += 1 / (powf(d_constraints[3] - stateInThisThreads[0],2) * invBarrier);
            if(stateInThisThreads[0] > d_constraints[3]){
                qx += 1000000;
            }
        }

        total_cost += qx;

        qx = 0.0f;

    }
    if(isnan(total_cost))
    {
        total_cost = 1000000 * 5;
    }
    float KL_COST, S, lambda, HM_COST, HM;
    //int NomarizationCost = sizeOfParaboloidElements + addTermForLSM; //LSMでparaboloidをフィッティングする際に行列の
    lambda = 2 * HORIZON;
    HM = total_cost / (0.1*HORIZON); //0.75
    S = total_cost / lambda;
    KL_COST = exp(-S);
    HM_COST = exp(-HM);
    __syncthreads();
    d_data[id].WHM = HM_COST;
    d_data[id].W = KL_COST;
    d_data[id].L = total_cost / sizeOfParaboloidElements;
    //d_data[id].L = total_cost;
    cost_vec[id] = total_cost;
    for(int index = 0; index < HORIZON; index++){
        d_data[id].Input[index] = u[index];
    }
    __syncthreads();

}