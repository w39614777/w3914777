#include "function.h"
#include "tools.h"
#include <queue>
int main(int argc ,char* argv[]){
    highprecision *eta1,*eta2,*eta1_lap,*eta2_lap,*dfdeta1,*dfdeta2,*eta1_out,*eta2_out;
    CHECK_ERROR(cudaMallocManaged((void**)&eta1,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta1_out,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2_out,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta1_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dfdeta1,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dfdeta2,sizeof(highprecision)*dimX*dimY));   
    lowprecision *heta1,*heta2,*heta1_lap,*heta2_lap,*hdfdeta1,*hdfdeta2,*heta1_out,*heta2_out;
    int *type_curr;
    CHECK_ERROR(cudaMallocManaged((void**)&heta1,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&heta2,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&heta1_out,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&heta2_out,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&heta1_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&heta2_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hdfdeta1,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hdfdeta2,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&type_curr,sizeof(int)*unitdimX*unitdimY));
    highprecision *eta2_last;
    CHECK_ERROR(cudaMallocManaged((void**)&eta2_last,sizeof(highprecision)*dimX*dimY));
    #ifdef AMSTENCIL
        highprecision *max_diff;
        CHECK_ERROR(cudaMallocManaged((void**)&max_diff,sizeof(highprecision)*unitdimX*unitdimY));
        #define TOLOW __float2half2_rn
        dim3 blocks_detect(unitdimX>32?32:unitdimX,unitdimY>32?32:unitdimY);
        dim3 grids_detect(unitdimX>32?unitdimX/32:1,unitdimY>32?unitdimY/32:1);
    #else
        #define TOLOW __float2half
    #endif
    lowprecision hmobil=TOLOW(mobil);
    lowprecision hgrcoef=TOLOW(grcoef);
    lowprecision hdx=TOLOW(dx);
    lowprecision hdy=TOLOW(dy);
    lowprecision hdxdy=TOLOW(dxdy);
    lowprecision height=TOLOW(8.0);
    lowprecision hdtime=TOLOW(dtime);
    lowprecision hone=TOLOW(1.0);
    lowprecision htwo=TOLOW(2.0);
    dim3 blocks_prepare_high(unitx,unity);
    dim3 grids_prepare_high(1,1,unitdimX*unitdimY);
    dataprepare_high<<<grids_prepare_high,blocks_prepare_high>>>(eta1,eta2);
    cudaDeviceSynchronize();

    dim3 blocks_pure(unitx,unity);
    dim3 grids_pure(1,1,unitdimX*unitdimY);
    for(int i=0;i<timesteps;i++){
        kernel1_pure<<<grids_pure,blocks_pure>>>(eta1,eta2,eta1_lap,dfdeta1,eta1_out);
        kernel1_pure<<<grids_pure,blocks_pure>>>(eta2,eta1,eta2_lap,dfdeta2,eta2_out);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
        #if ((defined Monitor2)&&(defined AMSTENCIL))
            if(i==timesteps-10){
                motivation_monitor2_datasychr<<<grids_pure,blocks_pure>>>(eta2,eta2_last);
                cudaDeviceSynchronize();
            }
        #endif        
    }
    dim3 blocks_mix(uxd2,unity);
    dim3 grids_mix(2,1,unitdimX*unitdimY);
    highdata_to_low<<<grids_mix,blocks_mix>>>(eta1,heta1);
    highdata_to_low<<<grids_mix,blocks_mix>>>(eta2,heta2);
    #ifdef AMSTENCIL
        int highprecisionnum=unitdimX*unitdimY-unitdimX*unitdimY*ratio,sum=0;
        #ifdef Monitor1
            get_max_diff1<<<grids_detect,blocks_detect>>>(eta2,max_diff);
        #endif
        #ifdef Monitor2
            get_max_diff2<<<grids_detect,blocks_detect>>>(eta2_last,eta2,max_diff);
        #endif
        cudaDeviceSynchronize();
        
        int *index;
        index=new int[unitdimX*unitdimY];
        BubbleSort(max_diff,unitdimX*unitdimY,index);
        for(int j=0;j<unitdimX*unitdimY;j++)type_curr[j]=1;
        queue<int> queue_index;
        for(int j=0;max_diff[j]!=0;j++){
            type_curr[index[j]]=2;
            sum++;
            queue_index.push(index[j]);
            if(sum>=highprecisionnum)break;
        }
        while(!queue_index.empty()){
            int center_index=queue_index.front();
            queue_index.pop();
            for(int direct=1;direct<=8;direct++){
                if(type_curr[get_neibour(center_index,direct,1)]!=2){
                    sum++;
                    type_curr[get_neibour(center_index,direct,1)]=2;
                    queue_index.push(get_neibour(center_index,direct,1));
                    if(sum>=highprecisionnum)break;
                }
            }
            if(sum>=highprecisionnum)break;
        }
        delete []index;
    #endif
    for(int i=timesteps;i<timesteps+50;i++){
        kernel1_mix<<<grids_mix,blocks_mix>>>(eta1,eta2_last,eta2,eta1_lap,dfdeta1,eta1_out,heta1,heta2,heta1_lap,hdfdeta1,heta1_out,height,hdxdy,hone,htwo,hdtime,hmobil,hgrcoef,type_curr,i,1);
        kernel1_mix<<<grids_mix,blocks_mix>>>(eta2,eta2_last,eta1,eta2_lap,dfdeta2,eta2_out,heta2,heta1,heta2_lap,hdfdeta2,heta2_out,height,hdxdy,hone,htwo,hdtime,hmobil,hgrcoef,type_curr,i,2);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
        swap(heta1,heta1_out);
        swap(heta2,heta2_out);

    }
    data_sychro_aftercomputation<<<grids_mix,blocks_mix>>>(eta2,heta2,type_curr);
    cudaDeviceSynchronize();
    string paths=string(argv[1]);
    writetocsv(paths,eta2,dimX,dimY);

    CHECK_ERROR(cudaFree(eta1));
    CHECK_ERROR(cudaFree(eta2));
    CHECK_ERROR(cudaFree(eta1_out));
    CHECK_ERROR(cudaFree(eta2_out));
    CHECK_ERROR(cudaFree(eta1_lap));
    CHECK_ERROR(cudaFree(eta2_lap));
    CHECK_ERROR(cudaFree(dfdeta1));
    CHECK_ERROR(cudaFree(dfdeta2));
    CHECK_ERROR(cudaFree(heta1));
    CHECK_ERROR(cudaFree(heta2));
    CHECK_ERROR(cudaFree(heta1_out));
    CHECK_ERROR(cudaFree(heta2_out));
    CHECK_ERROR(cudaFree(heta1_lap));
    CHECK_ERROR(cudaFree(heta2_lap));
    CHECK_ERROR(cudaFree(hdfdeta1));
    CHECK_ERROR(cudaFree(hdfdeta2));
    CHECK_ERROR(cudaFree(type_curr));
    CHECK_ERROR(cudaFree(eta2_last));
    #ifdef AMSTENCIL
        CHECK_ERROR(cudaFree(max_diff));
    #endif
    return 0;
}