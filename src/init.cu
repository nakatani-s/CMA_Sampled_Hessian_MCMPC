/*
 initialize all parameter for System model and cost function
*/ 
#include "../include/init.cuh"

void init_params( float *a )
{
    // params for simple nonlinear systems
    // for Simple Nonlinear System
    /*a[0] = 1.0f;
    a[1] = 1.0f;*/

    // FOR CART AND POLE
    a[0] = 0.1f;
    a[1] = 0.024f;
    a[2] = 0.2f;
    a[3] = a[1] * powf(a[2],2) /3;
    a[4] = 1.265f;
    a[5] = 0.0000001;
    a[6] = 9.81f;
}

void init_state( float *a )
{
    // initial state for simple nonlinear system
    // for Simple Nonlinear model
    /*a[0] = 2.0f;
    a[1] = 0.0f;*/
    
    // FOR CART AND POLE
    a[0] = 0.0f; //x
    a[1] = M_PI; //theta
    a[2] = 0.0f; //dx
    a[3] = 0.0f; //dth
}

void init_constraint( float *a )
{
    // constraints for simple nonlinear system
    // For Simple Nonlinera System
    /*a[0] = 0.0f;
    a[1] = 0.0f;*/

    // FOR CONTROL CART AND POLE
    a[0] = -1.0f;
    a[1] = 1.0f;
    a[2] = -0.45f;
    a[3] = 0.45f;
}

void init_matrix( float *a )
{
    //matrix elements for simple nonlinear system
    // FOR SIMPLE NONLINEAR SYSTEM
    /*a[0] = 0.1f;
    a[1] = 1.0f;
    a[2] = 0.1f;*/

    // FOR CAONTROL CART AND POLE
    a[0] = 3.0f;
    a[1] = 3.0f;
    a[2] = 0.04f;
    a[3] = 0.05f;
    a[4] = 0.5f;
}


void initialize_host_vector(float *get_params, float *get_state, float *get_constraints, float *get_matrix)
{
    init_params( get_params );
    init_state( get_state );
    init_constraint( get_constraints );
    init_matrix( get_matrix );
}
