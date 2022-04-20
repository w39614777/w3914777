#include "function.h"
#include "tools.h"
int main(void){
    highprecision *phi,*phi_lap,*tempr,*tempr_lap,*phidx,*phidy,*epsilon,*epsilon_deri;
    CHECK_ERROR(cudaMallocManaged((void**)&phi,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phi_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&tempr,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&tempr_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phidx,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phidy,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&epsilon,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&epsilon_deri,sizeof(highprecision)*dimX*dimY));
    lowprecision *hphi,*hphi_lap,*htempr,*htempr_lap,*hphidx,*hphidy,*hepsilon,*hepsilon_deri;
    int *type_new;
    CHECK_ERROR(cudaMallocManaged((void**)&hphi,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hphi_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&htempr,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&htempr_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hphidx,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hphidy,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hepsilon,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hepsilon_deri,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&type_new,sizeof(int)*unitdimX*unitdimY));
    highprecision *phi_old;
    CHECK_ERROR(cudaMallocManaged((void**)&phi_old,sizeof(highprecision)*dimX*dimY));
    #ifdef AMSTENCIL
        for(int i=0;i<unitdimX*unitdimY;i++){
            // type_new[i]=2;
            if(i<unitNums*ratio)type_new[i]=1;
            else type_new[i]=2;
        }
    #endif  
        dim3 blocks_preparehigh(unitx,unity);
        dim3 grids_preparehigh(1,1,unitdimX*unitdimY);
        dataprepare_high<<<grids_preparehigh,blocks_preparehigh>>>(phi);  

     #ifdef AMSTENCIL
        dim3 blocks_prepare_half2(uxd2,unity);
        dim3 grids_prepare_half2(1,1,unitdimX*unitdimY);
        dataprepare_half2<<<grids_prepare_half2,blocks_prepare_half2>>>(hphi);
        cudaDeviceSynchronize();
        #define TOLOW __float2half2_rn

    #else
        dim3 blocks_prepare_half(unitx,unity);
        dim3 grids_prepare_half(1,1,unitdimX*unitdimY);
        dataprepare_half<<<grids_prepare_half,blocks_prepare_half>>>(hphi);
        cudaDeviceSynchronize();
        #define TOLOW __float2half
    #endif
    lowprecision hdxdy=TOLOW(dxdy);
    lowprecision hdym2=TOLOW(dy*2);
    lowprecision hdxm2=TOLOW(dx*2);
    lowprecision height=TOLOW(8.0);
    lowprecision htheta0=TOLOW(theta0);
    lowprecision haniso=TOLOW(aniso);
    lowprecision hdelta=TOLOW(delta);
    lowprecision hone=TOLOW(1.0);
    lowprecision hepsilonb=TOLOW(epsilonb);
    lowprecision halpha=TOLOW(alpha);
    lowprecision hpi=TOLOW(pi);
    lowprecision hgama=TOLOW(gama);
    lowprecision hkappa=TOLOW(kappa);
    lowprecision hzpf=TOLOW(0.5);
    lowprecision hteq=TOLOW(teq);
    lowprecision hdtime=TOLOW(dtime);
    lowprecision htau=TOLOW(tau);  
    dim3 blocks_mix(uxd2,unity);
    dim3 grids_mix(2,1,unitdimX*unitdimY);
    for(int i=0;i<timesteps;i++){
        kernel1_mix<<<grids_mix,blocks_mix>>>(phi,phi_old,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap,hphi,hphi_lap,htempr,htempr_lap,hphidx,hphidy,hepsilon,hepsilon_deri,hdxdy,htheta0,haniso,hone,hdxm2,hdym2,hdelta,hepsilonb,height,type_new,i);
        mix_kernel2<<<grids_mix,blocks_mix>>>(phi,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap,hphi,hphi_lap,hepsilon,hepsilon_deri,hphidx,hphidy,htempr,htempr_lap,hdym2,hdxm2,hgama,hteq,halpha,hpi,hone,hzpf,hdtime,htau,hkappa,type_new,i);
        cudaDeviceSynchronize();
    }
    // #ifdef AMSTENCIL
    //     dim3 blocks(uxd2,unity);
    //     dim3 grids(1,1,unitdimX*unitdimY);
    // #else
    //     dim3 blocks(unitx,unity);
    //     dim3 grids(1,1,unitdimX*unitdimY);
    // #endif
    // purelow2high_aftercomputing<<<grids,blocks>>>(hphi,phi);
    // cudaDeviceSynchronize();
    data_sychro_aftercomputation<<<grids_mix,blocks_mix>>>(phi,hphi,type_new);
    cudaDeviceSynchronize();
    #ifdef AMSTENCIL
        writetocsv("amstencil.csv",phi,dimX,dimY);
    #endif
    #ifdef GRAM1
        writetocsv("gram1.csv",phi,dimX,dimY);
    #endif
    #ifdef GRAM2
        writetocsv("gram2.csv",phi,dimX,dimY);
    #endif
    CHECK_ERROR(cudaFree(phi));
    CHECK_ERROR(cudaFree(phi_lap));
    CHECK_ERROR(cudaFree(tempr));
    CHECK_ERROR(cudaFree(tempr_lap));
    CHECK_ERROR(cudaFree(phidx));
    CHECK_ERROR(cudaFree(phidy));
    CHECK_ERROR(cudaFree(epsilon));
    CHECK_ERROR(cudaFree(epsilon_deri));
    CHECK_ERROR(cudaFree(hphi));
    CHECK_ERROR(cudaFree(hphi_lap));
    CHECK_ERROR(cudaFree(htempr));
    CHECK_ERROR(cudaFree(htempr_lap));
    CHECK_ERROR(cudaFree(hphidx));
    CHECK_ERROR(cudaFree(hphidy));
    CHECK_ERROR(cudaFree(hepsilon));
    CHECK_ERROR(cudaFree(hepsilon_deri));
    CHECK_ERROR(cudaFree(type_new));
    CHECK_ERROR(cudaFree(phi_old));
    return 0;
}
