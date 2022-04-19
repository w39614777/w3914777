#include "function.h"
#include "tools.h"
int main(void){
    purelowprecision *eta1,*eta2,*eta1_lap,*eta2_lap,*dfdeta1,*dfdeta2,*eta1_out,*eta2_out;
    CHECK_ERROR(cudaMallocManaged((void**)&eta1,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta1_out,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2_out,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta1_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dfdeta1,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dfdeta2,sizeof(half)*dimX*dimY));
    #ifdef HALF
        dim3 blocks(unitx,unity);
        dim3 grids(1,1,unitdimX*unitdimY);
        dataprepare_half<<<grids,blocks>>>(eta1,eta2);
        cudaDeviceSynchronize();
        #define TOLOW __float2half
        #define FILENAME "halfresult.csv"
    #else
        dim3 blocks(uxd2,unity);
        dim3 grids(1,1,unitdimX*unitdimY);
        dataprepare_half2<<<grids,blocks>>>(eta1,eta2);
        cudaDeviceSynchronize();
        #define TOLOW __float2half2_rn
        #define FILENAME "half2result.csv"
    #endif
    purelowprecision hmobil=TOLOW(mobil);
    purelowprecision hgrcoef=TOLOW(grcoef);
    purelowprecision hdx=TOLOW(dx);
    purelowprecision hdy=TOLOW(dy);
    purelowprecision hdxdy=TOLOW(dxdy);
    purelowprecision height=TOLOW(8.0);
    purelowprecision hdtime=TOLOW(dtime);
    purelowprecision hone=TOLOW(1.0);
    purelowprecision htwo=TOLOW(2.0);
    for(int i=0;i<timesteps;i++){
        kernel1_lowpure<<<grids,blocks>>>(eta1,eta2,eta1_lap,dfdeta1,eta1_out,hone,htwo,height,hdtime,hmobil,hgrcoef,hdxdy);
        kernel1_lowpure<<<grids,blocks>>>(eta2,eta1,eta2_lap,dfdeta2,eta2_out,hone,htwo,height,hdtime,hmobil,hgrcoef,hdxdy);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
    }
    float *feta2;
    CHECK_ERROR(cudaMallocManaged((void**)&feta2,sizeof(float)*dimX*dimY));
    purelow2high_aftercomputing<<<grids,blocks>>>(eta2,feta2);
    cudaDeviceSynchronize();
    writetocsv(FILENAME,feta2,dimX,dimY);
    cout<<FILENAME<<endl;
    CHECK_ERROR(cudaFree(eta1));
    CHECK_ERROR(cudaFree(eta2));
    CHECK_ERROR(cudaFree(eta1_out));
    CHECK_ERROR(cudaFree(eta2_out));
    CHECK_ERROR(cudaFree(eta1_lap));
    CHECK_ERROR(cudaFree(eta2_lap));
    CHECK_ERROR(cudaFree(dfdeta1));
    CHECK_ERROR(cudaFree(dfdeta2));
    CHECK_ERROR(cudaFree(feta2));
}