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
    int *type_old;
    highprecision *max_diff;
    CHECK_ERROR(cudaMallocManaged((void**)&type_old,sizeof(int)*unitdimX*unitdimY));
    CHECK_ERROR(cudaMallocManaged((void**)&max_diff,sizeof(highprecision)*unitdimX*unitdimY));
    dim3 blocks_prepare_high(unitx,unity);
    dim3 grids_prepare_high(1,1,unitdimX*unitdimY);
    dataprepare_high<<<grids_prepare_high,blocks_prepare_high>>>(phi);
    cudaDeviceSynchronize();
    dim3 blocks_prepare_half2(uxd2,unity);
    dim3 grids_prepare_half2(1,1,unitdimX*unitdimY);
    dataprepare_half2<<<grids_prepare_half2,blocks_prepare_half2>>>(hphi);
    cudaDeviceSynchronize();
    #define TOLOW __float2half2_rn
    #ifdef Monitor2
        for(int i=0;i<dimX*dimY;i++){
            phi_old[i]=phi[i]==0?1.0:0.0;
        }
    #endif
    dim3 blocks_detect(unitdimX>32?32:unitdimX,unitdimY>32?32:unitdimY);
    dim3 grids_detect(unitdimX>32?unitdimX/32:1,unitdimY>32?unitdimY/32:1);
    dim3 datasychronduring_blocks(uxd2,unity);
    dim3 datasychronduring_grids(1,1,unitdimX*unitdimY);
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
    #ifdef Monitor1
        get_max_diff1<<<grids_detect,blocks_detect>>>(phi,max_diff);
    #endif
    #ifdef Monitor2
        get_max_diff2<<<grids_detect,blocks_detect>>>(phi_old,phi,max_diff);
        monitor2_lastdata_store<<<grids_mix,blocks_mix>>>(phi,phi_old);
    #endif
    get_type<<<grids_detect,blocks_detect>>>(max_diff,type_old,type_new);
    cudaDeviceSynchronize();
    data_sychro_duringcomputation<<<datasychronduring_grids,datasychronduring_blocks>>>(phi,tempr,hphi,htempr,type_old,type_new);
    cudaDeviceSynchronize();
    for(int i=0;i<5;i++){
        kernel1_mix<<<grids_mix,blocks_mix>>>(phi,phi_old,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap,hphi,hphi_lap,htempr,htempr_lap,hphidx,hphidy,hepsilon,hepsilon_deri,hdxdy,htheta0,haniso,hone,hdxm2,hdym2,hdelta,hepsilonb,height,type_new,i);
        kernel1_conversion<<<grids_mix,blocks_mix>>>(epsilon,epsilon_deri,phidx,phidy,hphidx,hphidy,hepsilon,hepsilon_deri,type_new,i);
        mix_kernel2<<<grids_mix,blocks_mix>>>(phi,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap,hphi,hphi_lap,hepsilon,hepsilon_deri,hphidx,hphidy,htempr,htempr_lap,hdym2,hdxm2,hgama,hteq,halpha,hpi,hone,hzpf,hdtime,htau,hkappa,type_new,i);
        kernel2_conversion<<<grids_mix,blocks_mix>>>(phi,tempr,hphi,htempr,type_new,i);
        cudaDeviceSynchronize();                 
    }
    cudaEvent_t kernel_start,kernel_end;float elapsed_kernel=0;
    CHECK_ERROR(cudaEventCreate(&kernel_start));
    CHECK_ERROR(cudaEventCreate(&kernel_end));
    float total_time=0;
    for(int i=5;i<timesteps;i++){
        if(i%10==0){
            #ifdef Monitor1
                get_max_diff1<<<grids_detect,blocks_detect>>>(phi,max_diff);
            #endif
            #ifdef Monitor2
                get_max_diff2<<<grids_detect,blocks_detect>>>(phi_old,phi,max_diff);
                cudaDeviceSynchronize();
                monitor2_lastdata_store<<<grids_mix,blocks_mix>>>(phi,phi_old);
            #endif
            cudaDeviceSynchronize();
            get_type<<<grids_detect,blocks_detect>>>(max_diff,type_old,type_new);
            cudaDeviceSynchronize();
            data_sychro_duringcomputation<<<datasychronduring_grids,datasychronduring_blocks>>>(phi,tempr,hphi,htempr,type_old,type_new);
            cudaDeviceSynchronize();
        }
        CHECK_ERROR(cudaEventRecord(kernel_start,0));
        kernel1_mix<<<grids_mix,blocks_mix>>>(phi,phi_old,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap,hphi,hphi_lap,htempr,htempr_lap,hphidx,hphidy,hepsilon,hepsilon_deri,hdxdy,htheta0,haniso,hone,hdxm2,hdym2,hdelta,hepsilonb,height,type_new,i);
        cudaDeviceSynchronize();
        CHECK_ERROR(cudaEventRecord(kernel_end,0));
        CHECK_ERROR(cudaEventSynchronize(kernel_end));
        CHECK_ERROR(cudaEventElapsedTime(&elapsed_kernel,kernel_start,kernel_end));
        total_time+=elapsed_kernel;

        kernel1_conversion<<<grids_mix,blocks_mix>>>(epsilon,epsilon_deri,phidx,phidy,hphidx,hphidy,hepsilon,hepsilon_deri,type_new,i);
        cudaDeviceSynchronize();


        CHECK_ERROR(cudaEventRecord(kernel_start,0));
        mix_kernel2<<<grids_mix,blocks_mix>>>(phi,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap,hphi,hphi_lap,hepsilon,hepsilon_deri,hphidx,hphidy,htempr,htempr_lap,hdym2,hdxm2,hgama,hteq,halpha,hpi,hone,hzpf,hdtime,htau,hkappa,type_new,i);
        cudaDeviceSynchronize();
        CHECK_ERROR(cudaEventRecord(kernel_end,0));
        CHECK_ERROR(cudaEventSynchronize(kernel_end));
        CHECK_ERROR(cudaEventElapsedTime(&elapsed_kernel,kernel_start,kernel_end));
        total_time+=elapsed_kernel;
        
        kernel2_conversion<<<grids_mix,blocks_mix>>>(phi,tempr,hphi,htempr,type_new,i);
        cudaDeviceSynchronize();  
    }
    CHECK_ERROR(cudaEventDestroy(kernel_start));
    CHECK_ERROR(cudaEventDestroy(kernel_end));
    ofstream ftime("time_tmp.csv");
    ftime<<total_time;
    ftime.close();
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
    CHECK_ERROR(cudaFree(type_old));
    CHECK_ERROR(cudaFree(max_diff));
    return 0;
}