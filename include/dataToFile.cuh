#include <stdio.h>
#include "params.cuh"
#include "DataStructure.cuh"
// #include "dynamics.cuh"

void get_timeParam(int *tparam,int month, int day, int hour, int min, int step);
void write_Matrix_Information(float *EigenVector, float *Matrix, int *timeparam);