#include "function.h"
#include "tools.h"
int main(int argc,char* argv[]){
    highprecision *eta1,*eta2,*eta1_lap,*eta2_lap,*dfdeta1,*dfdeta2,*eta1_out,*eta2_out;
    CHECK_ERROR(cudaMallocManaged((void**)&eta1,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta1_out,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2_out,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta1_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dfdeta1,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dfdeta2,sizeof(highprecision)*dimX*dimY));
    dim3 blocks(unitx,unity);
    dim3 grids(1,1,unitdimX*unitdimY);
    dataprepare_high<<<grids,blocks>>>(eta1,eta2);
    cudaDeviceSynchronize();
    #ifdef Motivation
         timesteps=timesteps+50;
    #endif
    #ifdef End2end
        cudaEvent_t start,stop;float elapsed;
    #endif
    for(int i=0;i<timesteps;i++){
        #ifdef End2end
            if(i==5){
                CHECK_ERROR(cudaEventCreate(&start));
                CHECK_ERROR(cudaEventCreate(&stop));
                CHECK_ERROR(cudaEventRecord(start,0));
            }
        #endif
        kernel1_pure<<<grids,blocks>>>(eta1,eta2,eta1_lap,dfdeta1,eta1_out);
        kernel1_pure<<<grids,blocks>>>(eta2,eta1,eta2_lap,dfdeta2,eta2_out);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
    }
    #ifdef End2end
        if(timesteps>5){
            CHECK_ERROR(cudaEventRecord(stop,0));
            CHECK_ERROR(cudaEventSynchronize(stop));
            CHECK_ERROR(cudaEventElapsedTime(&elapsed,start,stop));
            CHECK_ERROR(cudaEventDestroy(start));
            CHECK_ERROR(cudaEventDestroy(stop));
        }
        ofstream ftime("time_tmp.csv");
        ftime<<elapsed;
        ftime.close();
    #endif
    #ifdef End2end
        if(string(argv[1])=="4"){
            string path=string(argv[2]);
            writetocsv(path,eta2,dimX,dimY);
        }
    #endif
    #ifdef Motivation
        string path=string(argv[1]);
        writetocsv(path,eta2,dimX,dimY);
    #endif
    CHECK_ERROR(cudaFree(eta1));
    CHECK_ERROR(cudaFree(eta2));
    CHECK_ERROR(cudaFree(eta1_out));
    CHECK_ERROR(cudaFree(eta2_out));
    CHECK_ERROR(cudaFree(eta1_lap));
    CHECK_ERROR(cudaFree(eta2_lap));
    CHECK_ERROR(cudaFree(dfdeta1));
    CHECK_ERROR(cudaFree(dfdeta2));
    return 0;
}