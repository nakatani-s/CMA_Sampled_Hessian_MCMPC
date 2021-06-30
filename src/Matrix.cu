/* Matrix.cu */
#include "../include/Matrix.cuh" 

void printMatrix(int m, int n, float*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
            //printf("%s[%d] = %f\n", name, row + col*lda, Areg);
        }
    }
}

unsigned int countBlocks(unsigned int a, unsigned int b) {
	unsigned int num;
	num = a / b;
	if (a < b || a % b > 0)
		num++;
	return num;
}

int count_negative_EigenValues( float *EigVec, int step)
{
    int counter = 0;
    for(int i = 0; i < step; i++)
    {
        if(EigVec[i] <= 0.0f){
            counter += 1;
        }
    }
    return counter;
}

__global__ void setup_Identity_Matrix(float *IdMat)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(threadIdx.x == blockIdx.x)
    {
        IdMat[id] = 1.0f;
        //values[threadIdx.x] = 1.0f;
    }else{
        IdMat[id] = 0.0f;
        //values[threadIdx.x] = 0.0f;
    }
    __syncthreads();
}

__global__ void setup_Identity_Matrix_overMaxThread(float *IdMat, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int id = iy * Ydimention + ix;
    if(ix == iy)
    {
        IdMat[id] = 1.0f;
        //values[threadIdx.x] = 1.0f;
    }else{
        IdMat[id] = 0.0f;
        //values[threadIdx.x] = 0.0f;
    }
    __syncthreads();
}
/*__global__ void make_Covariance_Matrix(float *dCov,float *bCov, float *mean, InputVector d_Datas, int *indecies, int sz_smpl)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    float elements;
    // int denominator = 1 / (sz_smpl - 1);
    float f_denominator = 0.0f;
    // /分母（重みの和）を計算　←　重み付き標本共分散行列を使用する際 
    for(int num = 0; num < sz_smpl; num++)
    {
        if(isnan(d_Datas[indecies[num]].W))
        {
            f_denominator += 0.0f;
        }else{
            f_denominator += d_Datas[indecies[num]].W; 
        }
    }
    float weight_now;
    for(int i = 0; i < sz_smpl; i++)
    {
        weight_now = d_Datas[indecies[i]].W / f_denominator;
        // w  * (OP(y)-C^T) を愚直に計算した。
        elements += weight_now * ( (d_Datas[indecies[i]].Input[threadIdx.x] - mean[threadIdx.x]) * (d_Datas[indecies[i]].Input[threadIdx.x] - mean[threadIdx.x]) - bCov[id]);
    }

    dCov[id] = elements;
    __syncthreads( );
}*/

__global__ void copy_device_Matrix(float *Out, float *In)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    Out[id] = In[id];
    __syncthreads();
}

__global__ void make_SqrtEigen_Diagonal_Matrix(float *DiagMat, float *VecEig)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(threadIdx.x == blockIdx.x)
    {
        if(VecEig[threadIdx.x] < 0.0f)
        {
            DiagMat[id] = 0.0f;
        }else{
            DiagMat[id] = sqrt(VecEig[threadIdx.x]);
        }
    }else{
        DiagMat[id] = 0.0f;
    }
    __syncthreads( ); 
}


__global__ void make_IEDM_allow_suddlePoint(float *DiagMat, float *VecEig)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(threadIdx.x == blockIdx.x)
    {
        if(VecEig[threadIdx.x] <= 0.0f)
        {
            DiagMat[id] = 1/VecEig[threadIdx.x];
            // DiagMat[id] = -0.0001f;
            // DiagMat[id] = -1 / sqrt(fabs(VecEig[threadIdx.x]));
            // DiagMat[id] = VecEig[threadIdx.x];
        }else{
            // DiagMat[id] = VecEig[threadIdx.x];
            // DiagMat[id] = 1 / sqrt(VecEig[threadIdx.x]);
            DiagMat[id] = 1 / VecEig[threadIdx.x];
        }
    }else{
        DiagMat[id] = 0.0f;
    }
    __syncthreads( );
}


__global__ void make_InverseEigen_Diagonal_Matrix(float *DiagMat, float *VecEig)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(threadIdx.x == blockIdx.x)
    {
        if(VecEig[threadIdx.x] <= 0.0f)
        {
            DiagMat[id] = 0.0f;
            // DiagMat[id] = -0.0001f;
            // DiagMat[id] = -1 / sqrt(fabs(VecEig[threadIdx.x]));
            // DiagMat[id] = 1 / VecEig[threadIdx.x];
            // DiagMat[id] = 0.001f;
        }else{
            // DiagMat[id] = VecEig[threadIdx.x];
            // DiagMat[id] = 1 / sqrt(VecEig[threadIdx.x]);
            DiagMat[id] = 1 / VecEig[threadIdx.x];
        }
    }else{
        DiagMat[id] = 0.0f;
    }
    __syncthreads( );
}

// A * B → B
__global__ void pwr_matrix_answerLater(float *A, float *B)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    int row_index, column_index;
    float pows = 0.0f;
    if(blockIdx.x == 0)
    {
        row_index = (int)blockDim.x * blockIdx.x;
    }/*else{
        row_index = ((int)blockDim.x * blockIdx.x) -1;
    }*/
    if(threadIdx.x == 0)
    {
        column_index = (int)blockDim.x * threadIdx.x;
    }/*else{
        column_index = ((int)blockDim.x * threadIdx.x) -1;
    }*/
    for(int k = 0; k < HORIZON; k++){
        //row[id] += A[column_index + k] * B[row_index + k];
        pows += A[column_index + k] * B[row_index + k];
    }
    __syncthreads();
    B[id] = pows;
    /*if(threadIdx.x == 0)
    {
        B[id] = row[id];
    }*/
}

__global__ void prod_MtarixByMatrix(float *A, float *B, float *C, int num)
{
    unsigned int xid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int yid = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = xid * num + yid;
    
    float x = 0.0f;
    if(xid < num && yid < num)
    {
        for(int i = 0; i < num; i++)
        {
            x += A[i * num + xid] * B[i * num + yid];
        }
        C[idx] = x;
    }
    __syncthreads();
}

__global__ void multiply_matrix(float *OutMatrix, float voc, float *InMatrix)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    OutMatrix[id] = voc * InMatrix[id];
    //printf("OutMatrix[%d] == %f = %f * %f\n",id, OutMatrix[id], voc, InMatrix[id]);
    __syncthreads();
}

__global__ void copy_inputSequences(InputVector *outInput, float *temp)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    for(int i = 0; i < HORIZON; i++){
        outInput[id].Input[i] = temp[i];
    }
    __syncthreads();
}