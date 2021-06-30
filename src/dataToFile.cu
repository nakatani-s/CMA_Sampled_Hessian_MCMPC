/*
ある行列の固有値、固有ベクトルを.txtファイルに書き込む関数
*/
#include "../include/dataToFile.cuh"

void get_timeParam(int *tparam,int month, int day, int hour, int min, int step)
{
    tparam[0] = month;
    tparam[1] = day;
    tparam[2] = hour;
    tparam[3] = min;
    tparam[4] = step;
}

/*void printMatrix(int m, int n, float*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
            //printf("%s[%d] = %f\n", name, row + col*lda, Areg);
        }
    }
}*/

void write_Matrix_Information(float *EigenVector, float *grad, float *Matrix, int *timeparam)
{
    FILE *fp;
    char filename_here[35];
    sprintf(filename_here,"Hessian_Info_%d%d_%d%d_%d_step.txt", timeparam[0], timeparam[1], timeparam[2], timeparam[3], timeparam[4]);
    fp = fopen(filename_here, "w");

    for(int row = 0; row < HORIZON; row++){
        fprintf(fp, "%f %f ", EigenVector[row], grad[row]);
        for(int col = 0; col < HORIZON; col++)
        {
            if(col == HORIZON -1)
            {
                fprintf(fp,"%f\n", Matrix[row + col * HORIZON]);
            }else{
                fprintf(fp, "%f ", Matrix[row + col * HORIZON]);
            }
        }
    }

    fclose(fp);
    
}

/*void write_fittingError_percentage(float *numerator, float denominator, int step)
{
    FILE *fp;
    char filenames[25];
    sprintf(filenames, "FittigErrors%d.txt", step);
    fp = fopen(filenames, "w");

    for(int column = 0; column < 9; column++)
    {
        if(column < 8){
            fprintf(fp, "%f ", numerator[column] / denominator );
        }else{
            fprintf(fp, "%f\n")
        }
        
    }
}*/