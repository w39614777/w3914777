#include "function.h"
#include "tools.h"
int main(int argc,char* argv[]){
    highprecision *phi,*phi_lap,*tempr,*tempr_lap,*phidx,*phidy,*epsilon,*epsilon_deri;
    CHECK_ERROR(cudaMallocManaged((void**)&phi,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phi_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&tempr,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&tempr_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phidx,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phidy,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&epsilon,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&epsilon_deri,sizeof(highprecision)*dimX*dimY));
    dim3 blocks(unitx,unity);
    dim3 grids(1,1,unitdimX*unitdimY);
    dataprepare_high<<<grids,blocks>>>(phi);
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
        kernel1_pure<<<grids,blocks>>>(phi,phi_lap,tempr,tempr_lap,phidx,phidy,epsilon,epsilon_deri);
        kernel2_pure<<<grids,blocks>>>(phi,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap);
        cudaDeviceSynchronize();
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
            writetocsv(path,phi,dimX,dimY);
        }
    #endif
    #ifdef Motivation
        string path=string(argv[1]);
        writetocsv(path,phi,dimX,dimY);
    #endif
    CHECK_ERROR(cudaFree(phi));
    CHECK_ERROR(cudaFree(phi_lap));
    CHECK_ERROR(cudaFree(tempr));
    CHECK_ERROR(cudaFree(tempr_lap));
    CHECK_ERROR(cudaFree(phidx));
    CHECK_ERROR(cudaFree(phidy));
    CHECK_ERROR(cudaFree(epsilon));
    CHECK_ERROR(cudaFree(epsilon_deri));
    return 0;
}