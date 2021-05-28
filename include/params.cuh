/*
params.cuh
*/ 
// experiment1 for simple_example from Ohtsuka's book
#include <math.h>
#ifndef PARAMS_CUH
#define PARAMS_CUH

/*#define TIME 400
#define ITERATIONS 1
#define HORIZON 20
#define DIM_OF_PARAMETERS 2
#define DIM_OF_STATES 2
#define NUM_OF_CONSTRAINTS 2
#define DIM_OF_WEIGHT_MATRIX 3
#define DIM_OF_INPUT 1*/

// For Control Cart and Single Pole
#define TIME 1000
#define ITERATIONS 2
#define HORIZON 30  //50
#define DIM_OF_PARAMETERS 7
#define DIM_OF_STATES 4
#define NUM_OF_CONSTRAINTS 4
#define DIM_OF_WEIGHT_MATRIX 5
#define DIM_OF_INPUT 1

/* NUM_OF_SAMPLES について HORIZON^2 * (2/3) + 10程度に設定  ←　これ以下だと、２次曲面フィットで特異行列を扱うことになる　*/
#define NUM_OF_SAMPLES 15000
#define NUM_OF_ELITES  14// [4+3ln(20)]/2程度を確保　（←CMA−ESの類推より）
#define THREAD_PER_BLOCKS 10


const int sizeOfParaboloidElements = 496; //1326, HORIZON 30 -> 496 + 254 HORIZON 35 -> 666 + 134
const int addTermForLSM = 5504; //sizeOfParaboloidElements + addTermForLSM = THREAD_PER_BLOCKSの定数倍になるように加算する項  4000 - sizeOfParaboloidElements くらい
const float neighborVar = 0.8;
const float interval = 0.01;
const float variance = 2.0;
const float invBarrier = 10000;

const float c_learning_rate = 1.0f;

// #define INVERSE_OPERATION_USING_EIGENVALUE_FOR_LSM
#define INVERSE_OPERATION_USING_EIGENVALUE
#define CMA

#define WRITE_MATRIX_INFORMATION
#endif
// const int 
