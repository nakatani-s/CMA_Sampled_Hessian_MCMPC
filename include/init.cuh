/*
initialize all param for dynamical systems and setting of MCMPC

*/
#include <math.h>
#include "params.cuh"

void initialize_host_vector(float *get_params, float *get_state, float *get_constraints, float *get_matrix);

/*void init_sys_param(float *get_param);
void Mat_sys_A(float *a);
void init_state(float *st);
void init_Weight_matrix(float * matrix);

void init_opt( float *opt );
void init_constraint( float *constraint );*/