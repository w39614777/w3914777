#include "function.h"
#include "tools.h"
#include <queue>
int main(int argc ,char* argv[]){
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
        highprecision *max_diff;
        CHECK_ERROR(cudaMallocManaged((void**)&max_diff,sizeof(highprecision)*unitdimX*unitdimY));
        #define TOLOW __float2half2_rn
        dim3 blocks_detect(unitdimX>32?32:unitdimX,unitdimY>32?32:unitdimY);
        dim3 grids_detect(unitdimX>32?unitdimX/32:1,unitdimY>32?unitdimY/32:1);
    #else
        #define TOLOW __float2half
    #endif
    dim3 blocks_prepare_high(unitx,unity);
    dim3 grids_prepare_high(1,1,unitdimX*unitdimY);
    dataprepare_high<<<grids_prepare_high,blocks_prepare_high>>>(phi);
    cudaDeviceSynchronize();
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
    dim3 blocks_pure(unitx,unity);
    dim3 grids_pure(1,1,unitdimX*unitdimY);
    for(int i=0;i<timesteps;i++){
        kernel1_pure<<<grids_pure,blocks_pure>>>(phi,phi_lap,tempr,tempr_lap,phidx,phidy,epsilon,epsilon_deri);
        cudaDeviceSynchronize();
        kernel2_pure<<<grids_pure,blocks_pure>>>(phi,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap);
        cudaDeviceSynchronize();
        #if ((defined Monitor2)&&(defined AMSTENCIL))
            if(i==timesteps-10){
                motivation_monitor2_datasychr<<<grids_pure,blocks_pure>>>(phi,phi_old);
                cudaDeviceSynchronize();
            }
        #endif        
    }
    dim3 blocks_mix(uxd2,unity);
    dim3 grids_mix(2,1,unitdimX*unitdimY);
    highdata_to_low<<<grids_mix,blocks_mix>>>(phi,hphi);
    highdata_to_low<<<grids_mix,blocks_mix>>>(tempr,htempr);
    #ifdef AMSTENCIL
        int highprecisionnum=unitdimX*unitdimY-unitdimX*unitdimY*ratio,sum=0;
        #ifdef Monitor1
            get_max_diff1<<<grids_detect,blocks_detect>>>(phi,max_diff);
        #endif
        #ifdef Monitor2
            get_max_diff2<<<grids_detect,blocks_detect>>>(phi_old,phi,max_diff);
        #endif
        cudaDeviceSynchronize();
        int *index;
        index=new int[unitdimX*unitdimY];
        BubbleSort(max_diff,unitdimX*unitdimY,index);
        for(int j=0;j<unitdimX*unitdimY;j++)type_new[j]=1;
        queue<int> queue_index;
        for(int j=0;max_diff[j]!=0;j++){
            type_new[index[j]]=2;
            sum++;
            queue_index.push(index[j]);
            if(sum>=highprecisionnum)break;
        }
        while(!queue_index.empty()){
            int center_index=queue_index.front();
            queue_index.pop();
            for(int direct=1;direct<=8;direct++){
                if(type_new[get_neibour(center_index,direct,1)]!=2){
                    sum++;
                    type_new[get_neibour(center_index,direct,1)]=2;
                    queue_index.push(get_neibour(center_index,direct,1));
                    if(sum>=highprecisionnum)break;
                }
            }
            if(sum>=highprecisionnum)break;
        }
        delete []index;
    #endif
    for(int i=timesteps;i<timesteps+50;i++){
        kernel1_mix<<<grids_mix,blocks_mix>>>(phi,phi_old,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap,hphi,hphi_lap,htempr,htempr_lap,hphidx,hphidy,hepsilon,hepsilon_deri,hdxdy,htheta0,haniso,hone,hdxm2,hdym2,hdelta,hepsilonb,height,type_new,i);
        mix_kernel2<<<grids_mix,blocks_mix>>>(phi,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap,hphi,hphi_lap,hepsilon,hepsilon_deri,hphidx,hphidy,htempr,htempr_lap,hdym2,hdxm2,hgama,hteq,halpha,hpi,hone,hzpf,hdtime,htau,hkappa,type_new,i);
        cudaDeviceSynchronize();

    }
    data_sychro_aftercomputation<<<grids_mix,blocks_mix>>>(phi,hphi,type_new);
    cudaDeviceSynchronize();
    string paths=string(argv[1]);
    writetocsv(paths,phi,dimX,dimY);
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
    #ifdef AMSTENCIL
        CHECK_ERROR(cudaFree(max_diff));
    #endif
    return 0;    
}