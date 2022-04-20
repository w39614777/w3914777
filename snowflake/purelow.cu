#include "function.h"
#include "tools.h"
int main(void){
    purelowprecision *phi,*phi_lap,*tempr,*tempr_lap,*phidx,*phidy,*epsilon,*epsilon_deri;
    CHECK_ERROR(cudaMallocManaged((void**)&phi,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phi_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&tempr,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&tempr_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phidx,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phidy,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&epsilon,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&epsilon_deri,sizeof(half)*dimX*dimY));
    #ifdef HALF
        dim3 blocks(unitx,unity);
        dim3 grids(1,1,unitdimX*unitdimY);
        dataprepare_half<<<grids,blocks>>>(phi);
        cudaDeviceSynchronize();
        #define TOLOW __float2half
        #define FILENAME "halfresult.csv"
    #else
        dim3 blocks(uxd2,unity);
        dim3 grids(1,1,unitdimX*unitdimY);
        dataprepare_half2<<<grids,blocks>>>(phi);
        cudaDeviceSynchronize();
        #define TOLOW __float2half2_rn
        #define FILENAME "half2result.csv"
    #endif
    purelowprecision hdxdy=TOLOW(dxdy);
    purelowprecision hdym2=TOLOW(dy*2);
    purelowprecision hdxm2=TOLOW(dx*2);
    purelowprecision height=TOLOW(8.0);
    purelowprecision htheta0=TOLOW(theta0);
    purelowprecision haniso=TOLOW(aniso);
    purelowprecision hdelta=TOLOW(delta);
    purelowprecision hone=TOLOW(1.0);
    purelowprecision hepsilonb=TOLOW(epsilonb);
    purelowprecision halpha=TOLOW(alpha);
    purelowprecision hpi=TOLOW(pi);
    purelowprecision hgama=TOLOW(gama);
    purelowprecision hkappa=TOLOW(kappa);
    purelowprecision hzpf=TOLOW(0.5);
    purelowprecision hteq=TOLOW(teq);
    purelowprecision hdtime=TOLOW(dtime);
    purelowprecision htau=TOLOW(tau);
    cudaEvent_t start,stop;float elapsed;
    for(int i=0;i<timesteps;i++){
        if(i==5){
            CHECK_ERROR(cudaEventCreate(&start));
            CHECK_ERROR(cudaEventCreate(&stop));
            CHECK_ERROR(cudaEventRecord(start,0));
        }
        kernel1_purelow<<<grids,blocks>>>(phi,phi_lap,tempr,tempr_lap,phidx,phidy,epsilon,epsilon_deri,hdxdy,htheta0,haniso,hone,hdxm2,hdym2,hdelta,hepsilonb,height);
        kernel2_purelow<<<grids,blocks>>>(phi,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap,hdym2,hdxm2,hgama,hteq,halpha,hpi,hone,hzpf,hdtime,htau,hkappa);
        cudaDeviceSynchronize();
    }
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
    float *fphi;
    CHECK_ERROR(cudaMallocManaged((void**)&fphi,sizeof(float)*dimX*dimY));
    purelow2high_aftercomputing<<<grids,blocks>>>(phi,fphi);
    cudaDeviceSynchronize();
    writetocsv(FILENAME,fphi,dimX,dimY);
    cout<<FILENAME<<endl;
    CHECK_ERROR(cudaFree(phi));
    CHECK_ERROR(cudaFree(phi_lap));
    CHECK_ERROR(cudaFree(tempr));
    CHECK_ERROR(cudaFree(tempr_lap));
    CHECK_ERROR(cudaFree(phidx));
    CHECK_ERROR(cudaFree(phidy));
    CHECK_ERROR(cudaFree(epsilon));
    CHECK_ERROR(cudaFree(epsilon_deri));
    CHECK_ERROR(cudaFree(fphi));
    return 0;
}