/*
*/
#include <curand_kernel.h>
#include "params.cuh"
#ifndef DATASTRUCTUE_CUH
#define DATASTRUCTUE_CUH

typedef struct{
    float L;
    float W;
    float WHM;
    float Input[HORIZON];
}InputVector;


typedef struct{
    // float Input[HORIZON];
    float tensor_vector[sizeOfParaboloidElements];
    float column_vector[sizeOfParaboloidElements];

}QuadHyperPlane;

#endif