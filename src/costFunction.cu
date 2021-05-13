/* 
#include "../include/costFunction.cuh"
*/ 

#include "../include/costFunction.cuh"

float calc_Cost_Simple_NonLinear_Example( float *inputSequences, float *stateValues, float *param, float *weightMatrix)
{
    float costValue = 0.0f;
    float qx = 0.0f;
    float stateHere[DIM_OF_STATES] = { };
    float dStateValue[DIM_OF_STATES] = { };
    for(int i = 0; i < DIM_OF_STATES; i++){
        stateHere[i] = stateValues[i];
    }

    for(int tm = 0; tm < HORIZON; tm++){
        calc_nonLinear_example(stateHere, inputSequences[tm], param, dStateValue);
        stateHere[0] = stateHere[0] + (interval * dStateValue[0]);
        stateHere[1] = stateHere[1] + (interval * dStateValue[1]);

        qx = stateHere[0] * stateHere[0] * weightMatrix[0] + stateHere[1] * stateHere[1] * weightMatrix[1] + inputSequences[tm] * inputSequences[tm] * weightMatrix[2];
        costValue += qx;
        qx = 0.0f;
    }
    return costValue;
}

float calc_Cost_Cart_and_SinglePole(float *inputSeq, float *stateVal, float *param, float *constraints, float *weightMatrix)
{
    float costValue = 0.0f;
    float stageCost = 0.0f;
    float stateHere[DIM_OF_STATES] = { };
    float dStateValue[DIM_OF_STATES] = { };
    
    for(int no = 0; no < DIM_OF_STATES; no++){
        stateHere[no] = stateVal[no];
    }
    for(int t = 0; t < HORIZON; t++){
        if(inputSeq[t] < constraints[0]){
            inputSeq[t] = constraints[0];
        }
        if(inputSeq[t] > constraints[1]){
            inputSeq[t] = constraints[1];
        }
        // まずは、オイラー積分（100Hz 40stepで倒立できるか）　→　0.4秒先まで予測
        // 問題が起きたら、0次ホールダーでやってみる、それでもダメならMPCの再設計
        /*dStateValue[0] = stateHere[2];
        dStateValue[1] = stateHere[3];
        dStateValue[2] = Cart_type_Pendulum_ddx(inputSeq[t], stateHere[0], stateHere[1], stateHere[2], stateHere[3], param); //ddx
        dStateValue[3] = Cart_type_Pendulum_ddtheta(inputSeq[t], stateHere[0], stateHere[1], stateHere[2], stateHere[3], param);
        stateHere[2] = stateHere[2] + (interval * dStateValue[2]);
        stateHere[3] = stateHere[3] + (interval * dStateValue[3]);
        stateHere[0] = stateHere[0] + (interval * dStateValue[0]);
        stateHere[1] = stateHere[1] + (interval * dStateValue[1]);*/
        for(int sec = 0; sec < 3; sec++){
            dStateValue[0] = stateHere[2];
            dStateValue[1] = stateHere[3];
            dStateValue[2] = Cart_type_Pendulum_ddx(inputSeq[t], stateHere[0], stateHere[1], stateHere[2], stateHere[3], param); //ddx
            dStateValue[3] = Cart_type_Pendulum_ddtheta(inputSeq[t], stateHere[0], stateHere[1], stateHere[2], stateHere[3], param);
            stateHere[2] = stateHere[2] + (interval * dStateValue[2]);
            stateHere[3] = stateHere[3] + (interval * dStateValue[3]);
            stateHere[0] = stateHere[0] + (interval * dStateValue[0]);
            stateHere[1] = stateHere[1] + (interval * dStateValue[1]);
        }

        while(stateHere[1] > M_PI)
            stateHere[1] -= (2 * M_PI);
        while(stateHere[1] < -M_PI)
            stateHere[1] += (2 * M_PI);

        // upper side: MATLAB　で使用している評価関数を参考    
        /* qx = stateInThisThreads[0] * stateInThisThreads[0] * d_matrix[0] + stateInThisThreads[1] * stateInThisThreads[1] * d_matrix[1]
            + u[t] * u[t] * d_matrix[3]; */
        stageCost = stateHere[0] * stateHere[0] * weightMatrix[0] + stateHere[1] * stateHere[1] * weightMatrix[1]
            + stateHere[2] * stateHere[2] * weightMatrix[2] + stateHere[3] * stateHere[3] * weightMatrix[3]
            + inputSeq[t] * inputSeq[t] * weightMatrix[4];
        
        // constraints described by Barrier Function Method
        if(stateHere[0] <= 0){
            stageCost += 1 / (powf(stateHere[0] - constraints[2],2) * invBarrier);
            if(stateHere[0] < constraints[2]){
                stageCost += 1000000;
            }
        }else{
            stageCost += 1 / (powf(constraints[3] - stateHere[0],2) * invBarrier);
            if(stateHere[0] > constraints[3]){
                stageCost += 1000000;
            }
        }

        costValue += stageCost;

        stageCost = 0.0f;
    }
    printf("SUccess Val == %f\n", costValue);
    return costValue; 
}