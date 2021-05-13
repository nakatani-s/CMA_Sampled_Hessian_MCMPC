/* 

*/
#include "../include/LSM_QuadHyperPlane.cuh"

unsigned int count_QHP_Parameters(int hor)
{
    int ans  = 0;
    ans = (int) (1/2)*(hor * hor + 3* hor + 2);
    return ans;
}
__global__ void LSM_QHP_make_tensor_vector(QuadHyperPlane *output, InputVector *input, int *indices)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    int next_indices = 0;
    for(int i = 0; i < HORIZON; i++){
        for(int j = i; j < HORIZON; j++){
            output[id].tensor_vector[next_indices] = input[indices[id]].Input[i] * input[indices[id]].Input[j] * input[indices[id]].WHM;
	    //output[id].tensor_vector[next_indices] = i * j;
            output[id].column_vector[next_indices] = input[indices[id]].L * input[indices[id]].Input[i] * input[indices[id]].Input[j]  * input[indices[id]].WHM;
            //output[id].column_vector[next_indices] = 3.0;
            next_indices += 1;
        }
    }
    for(int i = 0; i < HORIZON; i++){
        output[id].tensor_vector[next_indices] = input[indices[id]].Input[i]  * input[indices[id]].WHM;
        //output[id].tensor_vector[next_indices] = i;
        output[id].column_vector[next_indices] = input[indices[id]].L * input[indices[id]].Input[i]  * input[indices[id]].WHM;
        //output[id].column_vector[next_indices] = (1/2)*i;
        next_indices += 1;
    }
    output[id].tensor_vector[sizeOfParaboloidElements - 1] = 1.0f  * input[indices[id]].WHM;
    output[id].column_vector[sizeOfParaboloidElements - 1] = input[indices[id]].L  * input[indices[id]].WHM;
    __syncthreads();
}

__global__ void LSM_QHP_make_regular_matrix(float *outRmatrix, QuadHyperPlane *elements, int sumSet)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    outRmatrix[id] = 0.0f;
    //float temp_here = 0.0f;
    for(int index = 0; index < sumSet; index++){
        outRmatrix[id] += elements[index].tensor_vector[threadIdx.x] * elements[index].tensor_vector[blockIdx.x];
        //float temp_here +=  
    }
    //printf("id==%d, ans == %f\n", id, outRmatrix[id]);
    __syncthreads();
}

//block内thread数が飽和したので、新たに作成(2021.3.8)
__global__ void LSM_QHP_make_regular_matrix_over_ThreadPerBlockLimit(float *outRmatrix, QuadHyperPlane *elements, int sumSet, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int id = iy * Ydimention + ix;
    outRmatrix[id] = 0.0f;
    for(int index = 0; index < sumSet; index++){
        outRmatrix[id] += elements[index].tensor_vector[ix] * elements[index].tensor_vector[iy];
    }
    __syncthreads();
}

__global__ void LSM_QHP_make_regular_vector(float *outRvector, QuadHyperPlane *elements, int sumSet)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    outRvector[id] = 0.0f;
    for(int index = 0; index < sumSet; index++)
    {
        outRvector[id] += elements[index].column_vector[id];
    }
    __syncthreads();
}

__global__ void LSM_QHP_get_reslt_all_elements(float *outElements, float *inElements)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    outElements[id] = inElements[id];
    //printf("outElements[%d] == %f\n", id, outElements[id]);
    __syncthreads();
}

__global__ void LSM_QHP_get_Hessian_Result(float *outElements, float *inElements)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    float temp_here;
    /*if(blockIdx.x == 0){
        //outElements[id] = inElements[threadIdx.x];
        temp_here = inElements[threadIdx.x];
    }
    if(threadIdx.x==0){
        //outElements[id] = inElements[blockIdx.x];
        temp_here = inElements[blockIdx.x];
    }
    if(threadIdx.x * blockIdx.x != 0){
        // int i_id;
        // i_id = blockIdx.x + (HORIZON - 1) + (threadIdx.x - 1);
        //outElements[id] = inElements[blockIdx.x + (HORIZON - 1) + (threadIdx.x - 1)];
        
        temp_here = inElements[blockIdx.x + (HORIZON - 1) + (threadIdx.x - 1)];
    }*/
    int vect_id = blockIdx.x;
    if(threadIdx.x <= blockIdx.x){
		for(int t_id = 0; t_id < threadIdx.x; t_id++){
            int sum_a = t_id + 1;
			vect_id += (HORIZON - sum_a);
		}
        //outElements[id] = inElements[vect_id];
        temp_here = inElements[vect_id];
    }else{
        //outElements[id] = 0.0f;
        temp_here = 0.0f;
    }
    if(threadIdx.x != blockIdx.x){
        //outElements[id] = outElements[id] / 2;
        outElements[id] = temp_here / 2;
    }else{
        outElements[id] = temp_here;
    }
    //printf("outElements[%d] == %f\n", id, outElements[id]);
    __syncthreads();
}

__global__ void LSM_QHP_transpose(float *Out, float *In)
{
	unsigned int id =threadIdx.x + blockIdx.x * blockDim.x;
    int In_index = blockIdx.x + threadIdx.x * blockDim.x;
    Out[id] = In[In_index];
    __syncthreads();
}

__global__ void LSM_QHP_make_symmetric(float *Out, float *In)
{
	unsigned int id =threadIdx.x + blockIdx.x * blockDim.x;
	if( blockIdx.x > threadIdx.x)
    {
		if(!(Out[id]==In[id]))
		{
			Out[id] = In[id];
		}
	}
}

__global__ void LSM_Hessian_To_Positive_Symmetric(float *Hess)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadIdx.x == blockIdx.x){
        Hess[id] += 0.5f;
    }
    __syncthreads();
}

/*  DAT-Methodで使用するAns = -2 * G^T * Hessian * Hvectorの　Hvectorを計算  */
__global__ void LSM_QHP_make_Hvector(float *Output, InputVector *datas, float *Hess)
{
    unsigned int id =threadIdx.x + blockIdx.x * blockDim.x;
    float power_vec_u1[HORIZON] = { };
    float power_vec_um[HORIZON] = { };
    float squareTerm_u1 = 0.0f;
    float squareTerm_um = 0.0f;

    for(int i = 0; i < HORIZON; i++){
        for(int k = 0; k < HORIZON; k++){
            power_vec_u1[i] += datas[0].Input[k] * Hess[i*HORIZON + k];
            power_vec_um[i] += datas[id+1].Input[k] * Hess[i*HORIZON + k];
        }
        squareTerm_u1 += power_vec_u1[i] * datas[0].Input[i];
        squareTerm_um += power_vec_um[i] * datas[id+1].Input[i];
    }
    Output[id] = datas[0].L - datas[id + 1].L - squareTerm_u1 + squareTerm_um;
    //printf("Output[%d] == %f  %f %f %f %f\n",id,Output[id],  datas[0].L, datas[id + 1].L, squareTerm_u1, squareTerm_um);
    __syncthreads();
}

/* DAT-Methodで使用するAns = -2 * G^T * Hessian * Hvectorの　-2*G^T行列を計算 */
__global__ void LSM_QHP_make_transGmatrix(float *Output, InputVector *datas)
{
    unsigned int id =threadIdx.x + blockIdx.x * blockDim.x;
    Output[id] = -2* (datas[0].Input[blockIdx.x] - datas[threadIdx.x + 1].Input[blockIdx.x]);
    __syncthreads();
} 

/* NAKニュートン法で使用するAns = -2 * Hessian * Hvectorの　Hvectorベクトルを計算 */
__global__ void LSM_QHP_make_bVector(float *OutVector, float *Elemets, int indecies)
{
    unsigned int id =threadIdx.x + blockIdx.x * blockDim.x;
    OutVector[id] = Elemets[indecies + id];
    //printf("id = %d IN = %f\n", indecies + id, Elemets[indecies + id]);
    __syncthreads();
}
