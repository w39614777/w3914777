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
        int *type_old;
        highprecision *max_diff;
        CHECK_ERROR(cudaMallocManaged((void**)&type_old,sizeof(int)*unitdimX*unitdimY));
        CHECK_ERROR(cudaMallocManaged((void**)&max_diff,sizeof(highprecision)*unitdimX*unitdimY));
    #endif
    dim3 blocks_prepare_high(unitx,unity);
    dim3 grids_prepare_high(1,1,unitdimX*unitdimY);
    dataprepare_high<<<grids_prepare_high,blocks_prepare_high>>>(eta1,eta2);
    cudaDeviceSynchronize();
    #ifdef AMSTENCIL
        dim3 blocks_prepare_half2(uxd2,unity);
        dim3 grids_prepare_half2(1,1,unitdimX*unitdimY);
        dataprepare_half2<<<grids_prepare_half2,blocks_prepare_half2>>>(heta1,heta2);
        cudaDeviceSynchronize();
        #define TOLOW __float2half2_rn
        #ifdef Monitor2
            
            for(int i=0;i<dimX*dimY;i++){
                eta2_last[i]=eta2[i]==0?1.0:0.0;
            }
        #endif
        dim3 blocks_detect(unitdimX>32?32:unitdimX,unitdimY>32?32:unitdimY);
        dim3 grids_detect(unitdimX>32?unitdimX/32:1,unitdimY>32?unitdimY/32:1);
        dim3 datasychronduring_blocks(uxd2,unity);
        dim3 datasychronduring_grids(1,1,unitdimX*unitdimY);
    #else
        dim3 blocks_prepare_half(unitx,unity);
        dim3 grids_prepare_half(1,1,unitdimX*unitdimY);
        dataprepare_half<<<grids_prepare_half,blocks_prepare_half>>>(heta1,heta2);
        cudaDeviceSynchronize();
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
    dim3 blocks_mix(uxd2,unity);
    dim3 grids_mix(2,1,unitdimX*unitdimY);
    cudaEvent_t startmix,stopmix;float elapsedmix;
    for(int i=0;i<timesteps;i++){
        if(i==5){
            CHECK_ERROR(cudaEventCreate(&startmix));
            CHECK_ERROR(cudaEventCreate(&stopmix));
            CHECK_ERROR(cudaEventRecord(startmix,0));
        }
        #ifdef AMSTENCIL
            if(i%10==0){
                #ifdef Monitor1
                get_max_diff1<<<grids_detect,blocks_detect>>>(eta2,max_diff);
                #endif
                #ifdef Monitor2
                get_max_diff2<<<grids_detect,blocks_detect>>>(eta2_last,eta2,max_diff);
                #endif
                get_type<<<grids_detect,blocks_detect>>>(max_diff,type_old,type_curr);
                cudaDeviceSynchronize();
                // writetocsv("max"+to_string(i)+".csv",max_diff,unitdimX,unitdimY);
                // writetocsv("type"+to_string(i)+".csv",type_curr,unitdimX,unitdimY);
                data_sychro_duringcomputation<<<datasychronduring_grids,datasychronduring_blocks>>>(eta1,eta2,heta1,heta2,type_old,type_curr);
                cudaDeviceSynchronize();
            }
        #endif
        kernel1_mix<<<grids_mix,blocks_mix>>>(eta1,eta2_last,eta2,eta1_lap,dfdeta1,eta1_out,heta1,heta2,heta1_lap,hdfdeta1,heta1_out,height,hdxdy,hone,htwo,hdtime,hmobil,hgrcoef,type_curr,i,1);
        kernel1_mix<<<grids_mix,blocks_mix>>>(eta2,eta2_last,eta1,eta2_lap,dfdeta2,eta2_out,heta2,heta1,heta2_lap,hdfdeta2,heta2_out,height,hdxdy,hone,htwo,hdtime,hmobil,hgrcoef,type_curr,i,2);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
        swap(heta1,heta1_out);
        swap(heta2,heta2_out);
    }
    if(timesteps>5){
        CHECK_ERROR(cudaEventRecord(stopmix,0));
        CHECK_ERROR(cudaEventSynchronize(stopmix));
        CHECK_ERROR(cudaEventElapsedTime(&elapsedmix,startmix,stopmix));
        CHECK_ERROR(cudaEventDestroy(startmix));
        CHECK_ERROR(cudaEventDestroy(stopmix));
    }
    ofstream ftime("time_tmp.csv");
    ftime<<elapsedmix;
    ftime.close();
    //每个程序跑五次测试性能，最后一次记录计算结果
    if(string(argv[1])=="4"){
        string path=string(argv[2]);
        data_sychro_aftercomputation<<<grids_mix,blocks_mix>>>(eta2,heta2,type_curr);
        cudaDeviceSynchronize();
        writetocsv(path,eta2,dimX,dimY);
    }
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
        CHECK_ERROR(cudaFree(type_old));
        CHECK_ERROR(cudaFree(max_diff));
    #endif
}