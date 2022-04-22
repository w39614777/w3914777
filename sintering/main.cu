#include "tools.h"
#include <sys/stat.h> 　
#include <sys/types.h>
#include<queue>
int main(int argc,char* argv[]){
    highprecision *con,*eta1,*eta2,*eta1_lap,*eta2_lap,*con_lap,*dummy,*dummy_lap,*dfdcon,*dfdeta1,*dfdeta2,*eta1_out,*eta2_out;
    CHECK_ERROR(cudaMallocManaged((void**)&con,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta1,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta1_out,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2_out,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dummy,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&con_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta1_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dummy_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dfdcon,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dfdeta1,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dfdeta2,sizeof(highprecision)*dimX*dimY));
    #if((defined AMSTENCIL)||(defined GRAM1)||(defined GRAM2))
    lowprecision *hcon,*heta1,*heta2,*heta1_lap,*heta2_lap,*hcon_lap,*hdummy,*hdummy_lap,*hdfdcon,*hdfdeta1,*hdfdeta2,*heta1_out,*heta2_out;
    int *type_con;
    CHECK_ERROR(cudaMallocManaged((void**)&hcon,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&heta1,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&heta2,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&heta1_out,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&heta2_out,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hdummy,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hcon_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&heta1_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&heta2_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hdummy_lap,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hdfdcon,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hdfdeta1,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&hdfdeta2,sizeof(half)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&type_con,sizeof(int)*unitdimX*unitdimY));
    #endif
    #ifdef AMSTENCIL
    int *type_old;
    highprecision *max_diff_con;
    CHECK_ERROR(cudaMallocManaged((void**)&type_old,sizeof(int)*unitdimX*unitdimY));
    CHECK_ERROR(cudaMallocManaged((void**)&max_diff_con,sizeof(highprecision)*unitdimX*unitdimY));
    #endif

    for(int y=1;y<=dimY;y++){
        for(int x=1;x<=dimX;x++){
            float dis1=sqrt(pow(x-Rx1,2)+pow(y-Ry1,2));
            float dis2=sqrt(pow(x-Rx1,2)+pow(y-Ry2,2));
            if(dis1<=R1){
                con[(y-1)*dimX+x-1]=1;
                eta1[(y-1)*dimX+x-1]=1;
            }
            if(dis2<=R2){
                con[(y-1)*dimX+x-1]=1;
                eta2[(y-1)*dimX+x-1]=1;
                eta1[(y-1)*dimX+x-1]=0.0;
            }
        }
    }
    #if ((defined AMSTENCIL)||(defined GRAM1||(defined GRAM2)))
    highprecision *con_old;
    CHECK_ERROR(cudaMallocManaged((void**)&con_old,sizeof(highprecision)*dimX*dimY));
    for(int i=0;i<dimX*dimY;i++){
        con_old[i]=con[i]==0?1.0:0.0;
    }
    #endif
    #if (defined AMSTENCIL)&&(defined Motivation)&&(defined Monitor2)
    highprecision *con_last;
    CHECK_ERROR(cudaMallocManaged((void**)&con_last,sizeof(highprecision)*dimX*dimY));
    #endif 
    #if ((defined AMSTENCIL)||(defined GRAM1||(defined GRAM2)))
    dim3 blocks_dataprepare(32,32);
    dim3 grids_dataprepare(dimX/32/2,dimY/32);
    dataprepare<<<grids_dataprepare,blocks_dataprepare>>>(con,hcon);
    dataprepare<<<grids_dataprepare,blocks_dataprepare>>>(eta1,heta1);
    dataprepare<<<grids_dataprepare,blocks_dataprepare>>>(eta2,heta2);
    cudaDeviceSynchronize();
    lowprecision hcoefm=lowprecision2highprecision(coefm);
    lowprecision hcoefk=lowprecision2highprecision(coefk);
    lowprecision hcoefl=lowprecision2highprecision(coefl);
    lowprecision hdvol=lowprecision2highprecision(dvol);
    lowprecision hdvap=lowprecision2highprecision(dvap);
    lowprecision hdsur=lowprecision2highprecision(dsur);
    lowprecision hdgrb=lowprecision2highprecision(dgrb);
    lowprecision hdx=lowprecision2highprecision(dx);
    lowprecision hdy=lowprecision2highprecision(dy);
    lowprecision hdxdy=lowprecision2highprecision(dxdy);
    lowprecision hdtime=lowprecision2highprecision(dtime);
    lowprecision hcf=lowprecision2highprecision(4.0);
    lowprecision hzpfive=lowprecision2highprecision(0.5);
    lowprecision hone=lowprecision2highprecision(1.0);
    lowprecision htwo=lowprecision2highprecision(2.0);
    lowprecision hfour=lowprecision2highprecision(4.0);
    lowprecision hsix=lowprecision2highprecision(6.0);
    lowprecision hten=lowprecision2highprecision(10.0);
    lowprecision htwelve=lowprecision2highprecision(12.0);
    lowprecision hfifteen=lowprecision2highprecision(15.0);
    lowprecision hthirtytwo=lowprecision2highprecision(32.0);
    lowprecision highbound=lowprecision2highprecision(1);
    lowprecision lowbound=lowprecision2highprecision(0.0);
    #endif

    // 线程数量设置
    #if  ((defined PURE)||(defined Motivation))
    dim3 blocks_pure(unitx,unity);
    dim3 grids_pure(1,1,unitdimX*unitdimY);
    #endif
    #if ((defined AMSTENCIL)||(defined GRAM1||(defined GRAM2))) 
    dim3 blocks(uxd2,unity);
    dim3 grids(2,1,unitdimX*unitdimY);
    #endif
    #ifdef AMSTENCIL
    dim3 blocks_detect(unitdimX>32?32:unitdimX,unitdimY>32?32:unitdimY);
    dim3 grids_detect(unitdimX>32?unitdimX/32:1,unitdimY>32?unitdimY/32:1);
    dim3 datasychronduring_blocks(uxd2,unity);
    dim3 datasychronduring_grids(1,1,unitdimX*unitdimY);
    #endif
    // 计时
    #ifdef GET_TIME
    cudaEvent_t startmix,stopmix;float elapsedmix;
    #endif
    #ifdef End2end
    #ifdef GET_MonitorTIME
    float monitor_time=0;
    #endif
    for(int i=0;i<timesteps;i++){
        
        #ifdef GET_TIME
        if(i==5){
            CHECK_ERROR(cudaEventCreate(&startmix));
            CHECK_ERROR(cudaEventCreate(&stopmix));
            CHECK_ERROR(cudaEventRecord(startmix,0));
            CHECK_ERROR(cudaEventSynchronize(startmix));
        }
        #endif
        #ifdef PURE
        con1_pure<<<grids_pure,blocks_pure>>>(con,con_lap,eta1,eta2,dfdcon,dummy,i);
        cudaDeviceSynchronize();
        con2_pure<<<grids_pure,blocks_pure>>>(dummy,dummy_lap,con,eta1,eta2,i);
        cudaDeviceSynchronize();
        phi1_pure<<<grids_pure,blocks_pure>>>(eta1,eta1_out,eta2,eta1_lap,dfdeta1,con,i);
        cudaDeviceSynchronize();
        phi2_pure<<<grids_pure,blocks_pure>>>(eta2,eta2_out,eta1,eta2_lap,dfdeta2,con,i);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
        #endif
        #if((defined GRAM1)||(defined GRAM2))
        con1_mix<<<grids,blocks>>>(type_con,con,con_old,con_lap,eta1,eta2,dfdcon,dummy,hcon,hcon_lap,heta1,heta2,hdfdcon,hdummy,hcf,hdxdy,hsix,hzpfive,hone,htwo,hfour,hthirtytwo,hcoefm,i);
        cudaDeviceSynchronize();
        con2_mix<<<grids,blocks>>>(type_con,dummy,dummy_lap,con,eta1,eta2,hdummy,hdummy_lap,hcon,heta1,heta2,hcf,hdxdy,hone,htwo,hsix,hten,hfifteen,hdgrb,hdsur,hdtime,hdvap,hdvol,highbound,lowbound,i);
        cudaDeviceSynchronize();
        phi1_mix<<<grids,blocks>>>(type_con,eta1,eta1_out,eta2,eta1_lap,dfdeta1,con,heta1,heta1_out,heta2,heta1_lap,hdfdeta1,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        cudaDeviceSynchronize();
        phi2_mix<<<grids,blocks>>>(type_con,eta2,eta2_out,eta1,eta2_lap,dfdeta2,con,heta2,heta2_out,heta1,heta2_lap,hdfdeta2,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
        swap(heta1,heta1_out);
        swap(heta2,heta2_out);
        #endif
        #ifdef AMSTENCIL
        if(i%10==0){
            #ifdef GET_MonitorTIME
            cudaEvent_t startmonitor,stopmonitor;float elapsedmonitor=0;
            if(i>=5){
                CHECK_ERROR(cudaEventCreate(&startmonitor));
                CHECK_ERROR(cudaEventCreate(&stopmonitor));
                CHECK_ERROR(cudaEventRecord(startmonitor,0));
                CHECK_ERROR(cudaEventSynchronize(startmonitor));
            }
            #endif
            #ifdef Monitor1
            get_max_diff1<<<grids_detect,blocks_detect>>>(con,max_diff_con);
            #endif
            #ifdef Monitor2
            get_max_diff2<<<grids_detect,blocks_detect>>>(con_old,con,max_diff_con);
            #endif
            get_type<<<grids_detect,blocks_detect>>>(max_diff_con,type_old,type_con);
            cudaDeviceSynchronize();
            data_sychro_duringcomputation<<<datasychronduring_grids,datasychronduring_blocks>>>(con,eta1,eta2,hcon,heta1,heta2,type_old,type_con);
            cudaDeviceSynchronize();
            #ifdef GET_MonitorTIME
            if(i>=5){
                CHECK_ERROR(cudaEventRecord(stopmonitor,0));
                CHECK_ERROR(cudaEventSynchronize(stopmonitor));
                CHECK_ERROR(cudaEventElapsedTime(&elapsedmonitor,startmonitor,stopmonitor));
                CHECK_ERROR(cudaEventDestroy(startmonitor));
                CHECK_ERROR(cudaEventDestroy(stopmonitor));
            }
            monitor_time=monitor_time+elapsedmonitor;
            #endif
        }
        con1_mix<<<grids,blocks>>>(type_con,con,con_old,con_lap,eta1,eta2,dfdcon,dummy,hcon,hcon_lap,heta1,heta2,hdfdcon,hdummy,hcf,hdxdy,hsix,hzpfive,hone,htwo,hfour,hthirtytwo,hcoefm,i);
        cudaDeviceSynchronize();
        con2_mix<<<grids,blocks>>>(type_con,dummy,dummy_lap,con,eta1,eta2,hdummy,hdummy_lap,hcon,heta1,heta2,hcf,hdxdy,hone,htwo,hsix,hten,hfifteen,hdgrb,hdsur,hdtime,hdvap,hdvol,highbound,lowbound,i);
        cudaDeviceSynchronize();
        phi1_mix<<<grids,blocks>>>(type_con,eta1,eta1_out,eta2,eta1_lap,dfdeta1,con,heta1,heta1_out,heta2,heta1_lap,hdfdeta1,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        cudaDeviceSynchronize();
        phi2_mix<<<grids,blocks>>>(type_con,eta2,eta2_out,eta1,eta2_lap,dfdeta2,con,heta2,heta2_out,heta1,heta2_lap,hdfdeta2,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
        swap(heta1,heta1_out);
        swap(heta2,heta2_out);
        #endif
    }
    #ifdef GET_MonitorTIME
    ofstream ftime("time_tmp.csv");
    ftime<<monitor_time;
    ftime.close();
    #endif
    #endif
    #ifdef GET_TIME
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
    #endif
    #ifdef Motivation
    for(int i=0;i<timesteps;i++){
        con1_pure<<<grids_pure,blocks_pure>>>(con,con_lap,eta1,eta2,dfdcon,dummy,i);
        cudaDeviceSynchronize();
        con2_pure<<<grids_pure,blocks_pure>>>(dummy,dummy_lap,con,eta1,eta2,i);
        cudaDeviceSynchronize();
        phi1_pure<<<grids_pure,blocks_pure>>>(eta1,eta1_out,eta2,eta1_lap,dfdeta1,con,i);
        cudaDeviceSynchronize();
        phi2_pure<<<grids_pure,blocks_pure>>>(eta2,eta2_out,eta1,eta2_lap,dfdeta2,con,i);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out); 
        #if ((defined Monitor2)&&(defined AMSTENCIL))
        if(i==timesteps-10){
            for(int j=0;j<dimX*dimY;j++)con_last[j]=con[j];
        }
        #endif
    }
    for(int i=timesteps;i<timesteps+50;i++){
        #ifdef PURE
        con1_pure<<<grids_pure,blocks_pure>>>(con,con_lap,eta1,eta2,dfdcon,dummy,i);
        cudaDeviceSynchronize();
        con2_pure<<<grids_pure,blocks_pure>>>(dummy,dummy_lap,con,eta1,eta2,i);
        cudaDeviceSynchronize();
        phi1_pure<<<grids_pure,blocks_pure>>>(eta1,eta1_out,eta2,eta1_lap,dfdeta1,con,i);
        cudaDeviceSynchronize();
        phi2_pure<<<grids_pure,blocks_pure>>>(eta2,eta2_out,eta1,eta2_lap,dfdeta2,con,i);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out); 
        #endif
        #if((defined GRAM1)||(defined GRAM2))
        if(i==timesteps){
            dataprepare<<<grids_dataprepare,blocks_dataprepare>>>(con,hcon);
            dataprepare<<<grids_dataprepare,blocks_dataprepare>>>(eta1,heta1);
            dataprepare<<<grids_dataprepare,blocks_dataprepare>>>(eta2,heta2);
        }
        con1_mix<<<grids,blocks>>>(type_con,con,con_old,con_lap,eta1,eta2,dfdcon,dummy,hcon,hcon_lap,heta1,heta2,hdfdcon,hdummy,hcf,hdxdy,hsix,hzpfive,hone,htwo,hfour,hthirtytwo,hcoefm,i);
        cudaDeviceSynchronize();
        con2_mix<<<grids,blocks>>>(type_con,dummy,dummy_lap,con,eta1,eta2,hdummy,hdummy_lap,hcon,heta1,heta2,hcf,hdxdy,hone,htwo,hsix,hten,hfifteen,hdgrb,hdsur,hdtime,hdvap,hdvol,highbound,lowbound,i);
        cudaDeviceSynchronize();
        phi1_mix<<<grids,blocks>>>(type_con,eta1,eta1_out,eta2,eta1_lap,dfdeta1,con,heta1,heta1_out,heta2,heta1_lap,hdfdeta1,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        cudaDeviceSynchronize();
        phi2_mix<<<grids,blocks>>>(type_con,eta2,eta2_out,eta1,eta2_lap,dfdeta2,con,heta2,heta2_out,heta1,heta2_lap,hdfdeta2,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
        swap(heta1,heta1_out);
        swap(heta2,heta2_out);
        #endif
        #ifdef AMSTENCIL
        if(i==timesteps){
            int *index;
            index=new int[unitdimX*unitdimY];
            dataprepare<<<grids_dataprepare,blocks_dataprepare>>>(con,hcon);
            dataprepare<<<grids_dataprepare,blocks_dataprepare>>>(eta1,heta1);
            dataprepare<<<grids_dataprepare,blocks_dataprepare>>>(eta2,heta2);
            int highprecisionnum=unitdimX*unitdimY-unitdimX*unitdimY*ratio,sum=0;
            #ifdef Monitor1
            get_max_diff1<<<grids_detect,blocks_detect>>>(con,max_diff_con);
            #endif
            #ifdef Monitor2
            get_max_diff2<<<grids_detect,blocks_detect>>>(con_last,con,max_diff_con);
            #endif
            cudaDeviceSynchronize();
            BubbleSort(max_diff_con,unitdimX*unitdimY,index);
            for(int j=0;j<unitdimX*unitdimY;j++)type_con[j]=1;
            queue<int> queue_index;
            for(int j=0;max_diff_con[j]!=0;j++){
                type_con[index[j]]=2;
                sum++;
                queue_index.push(index[j]);
                if(sum>=highprecisionnum)break;
            }
            while(!queue_index.empty()){
                int center_index=queue_index.front();
                queue_index.pop();
                for(int direct=1;direct<=8;direct++){
                    if(type_con[get_neibour(center_index,direct,1)]!=2){
                        sum++;
                        type_con[get_neibour(center_index,direct,1)]=2;
                        queue_index.push(get_neibour(center_index,direct,1));
                        if(sum>=highprecisionnum)break;
                    }
                }
                if(sum>=highprecisionnum)break;
            }
            delete []index;
        }

        con1_mix<<<grids,blocks>>>(type_con,con,con_old,con_lap,eta1,eta2,dfdcon,dummy,hcon,hcon_lap,heta1,heta2,hdfdcon,hdummy,hcf,hdxdy,hsix,hzpfive,hone,htwo,hfour,hthirtytwo,hcoefm,i);
        cudaDeviceSynchronize();
        con2_mix<<<grids,blocks>>>(type_con,dummy,dummy_lap,con,eta1,eta2,hdummy,hdummy_lap,hcon,heta1,heta2,hcf,hdxdy,hone,htwo,hsix,hten,hfifteen,hdgrb,hdsur,hdtime,hdvap,hdvol,highbound,lowbound,i);
        cudaDeviceSynchronize();
        phi1_mix<<<grids,blocks>>>(type_con,eta1,eta1_out,eta2,eta1_lap,dfdeta1,con,heta1,heta1_out,heta2,heta1_lap,hdfdeta1,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        cudaDeviceSynchronize();
        phi2_mix<<<grids,blocks>>>(type_con,eta2,eta2_out,eta1,eta2_lap,dfdeta2,con,heta2,heta2_out,heta1,heta2_lap,hdfdeta2,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
        swap(heta1,heta1_out);
        swap(heta2,heta2_out);
        #endif
    }
    #endif

    #ifdef monitor_conversion_independent
    #ifdef Monitor1
    get_max_diff1<<<grids_detect,blocks_detect>>>(con,max_diff_con);
    #endif
    #ifdef Monitor2
    get_max_diff2<<<grids_detect,blocks_detect>>>(con_old,con,max_diff_con);
    #endif
    get_type<<<grids_detect,blocks_detect>>>(max_diff_con,type_old,type_con);
    cudaDeviceSynchronize();
    data_sychro_duringcomputation<<<datasychronduring_grids,datasychronduring_blocks>>>(con,eta1,eta2,hcon,heta1,heta2,type_old,type_con);
    cudaDeviceSynchronize();
    for(int i=0;i<5;i++){
        #ifdef Monitor2
        monitor2_lastdata_store<<<grids,blocks>>>(con,con_old,i);
        cudaDeviceSynchronize();
        #endif

        con1_mix<<<grids,blocks>>>(type_con,con,con_old,con_lap,eta1,eta2,dfdcon,dummy,hcon,hcon_lap,heta1,heta2,hdfdcon,hdummy,hcf,hdxdy,hsix,hzpfive,hone,htwo,hfour,hthirtytwo,hcoefm,i);
        cudaDeviceSynchronize();
        con1_conversion<<<grids,blocks>>>(type_con,dummy,hdummy);
        cudaDeviceSynchronize();

        con2_mix<<<grids,blocks>>>(type_con,dummy,dummy_lap,con,eta1,eta2,hdummy,hdummy_lap,hcon,heta1,heta2,hcf,hdxdy,hone,htwo,hsix,hten,hfifteen,hdgrb,hdsur,hdtime,hdvap,hdvol,highbound,lowbound,i);
        cudaDeviceSynchronize();
        con2_conversion<<<grids,blocks>>>(type_con,con,hcon,i);
        cudaDeviceSynchronize();

        phi1_mix<<<grids,blocks>>>(type_con,eta1,eta1_out,eta2,eta1_lap,dfdeta1,con,heta1,heta1_out,heta2,heta1_lap,hdfdeta1,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        cudaDeviceSynchronize();
        phi1_conversion<<<grids,blocks>>>(type_con,eta1_out,heta1_out,i);
        cudaDeviceSynchronize();

        phi2_mix<<<grids,blocks>>>(type_con,eta2,eta2_out,eta1,eta2_lap,dfdeta2,con,heta2,heta2_out,heta1,heta2_lap,hdfdeta2,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        phi2_conversion<<<grids,blocks>>>(type_con,eta2_out,heta2_out,i);
        cudaDeviceSynchronize();

        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
        swap(heta1,heta1_out);
        swap(heta2,heta2_out);
    }
    cudaEvent_t kernel_start,kernel_end;float elapsed_kernel=0;
    CHECK_ERROR(cudaEventCreate(&kernel_start));
    CHECK_ERROR(cudaEventCreate(&kernel_end));
    float total_time=0;
    for(int i=5;i<timesteps;i++){
        if(i%10==0){
            #ifdef Monitor1
            get_max_diff1<<<grids_detect,blocks_detect>>>(con,max_diff_con);
            #endif
            #ifdef Monitor2
            get_max_diff2<<<grids_detect,blocks_detect>>>(con_old,con,max_diff_con);
            #endif
            get_type<<<grids_detect,blocks_detect>>>(max_diff_con,type_old,type_con);
            cudaDeviceSynchronize();
            data_sychro_duringcomputation<<<datasychronduring_grids,datasychronduring_blocks>>>(con,eta1,eta2,hcon,heta1,heta2,type_old,type_con);
            cudaDeviceSynchronize();
        }
        #ifdef Monitor2
        monitor2_lastdata_store<<<grids,blocks>>>(con,con_old,i);
        cudaDeviceSynchronize();
        #endif

        CHECK_ERROR(cudaEventRecord(kernel_start,0));
        con1_mix<<<grids,blocks>>>(type_con,con,con_old,con_lap,eta1,eta2,dfdcon,dummy,hcon,hcon_lap,heta1,heta2,hdfdcon,hdummy,hcf,hdxdy,hsix,hzpfive,hone,htwo,hfour,hthirtytwo,hcoefm,i);
        cudaDeviceSynchronize();
        CHECK_ERROR(cudaEventRecord(kernel_end,0));
        CHECK_ERROR(cudaEventSynchronize(kernel_end));
        CHECK_ERROR(cudaEventElapsedTime(&elapsed_kernel,kernel_start,kernel_end));
        total_time+=elapsed_kernel;
        con1_conversion<<<grids,blocks>>>(type_con,dummy,hdummy);
        cudaDeviceSynchronize();

        CHECK_ERROR(cudaEventRecord(kernel_start,0));
        con2_mix<<<grids,blocks>>>(type_con,dummy,dummy_lap,con,eta1,eta2,hdummy,hdummy_lap,hcon,heta1,heta2,hcf,hdxdy,hone,htwo,hsix,hten,hfifteen,hdgrb,hdsur,hdtime,hdvap,hdvol,highbound,lowbound,i);
        cudaDeviceSynchronize();
        CHECK_ERROR(cudaEventRecord(kernel_end,0));
        CHECK_ERROR(cudaEventSynchronize(kernel_end));
        CHECK_ERROR(cudaEventElapsedTime(&elapsed_kernel,kernel_start,kernel_end));
        total_time+=elapsed_kernel;
        con2_conversion<<<grids,blocks>>>(type_con,con,hcon,i);
        cudaDeviceSynchronize();

        CHECK_ERROR(cudaEventRecord(kernel_start,0));
        phi1_mix<<<grids,blocks>>>(type_con,eta1,eta1_out,eta2,eta1_lap,dfdeta1,con,heta1,heta1_out,heta2,heta1_lap,hdfdeta1,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        cudaDeviceSynchronize();
        CHECK_ERROR(cudaEventRecord(kernel_end,0));
        CHECK_ERROR(cudaEventSynchronize(kernel_end));
        CHECK_ERROR(cudaEventElapsedTime(&elapsed_kernel,kernel_start,kernel_end));
        total_time+=elapsed_kernel;
        phi1_conversion<<<grids,blocks>>>(type_con,eta1_out,heta1_out,i);
        cudaDeviceSynchronize();

        CHECK_ERROR(cudaEventRecord(kernel_start,0));
        phi2_mix<<<grids,blocks>>>(type_con,eta2,eta2_out,eta1,eta2_lap,dfdeta2,con,heta2,heta2_out,heta1,heta2_lap,hdfdeta2,hcon,hcf,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound,i);
        CHECK_ERROR(cudaEventRecord(kernel_end,0));
        CHECK_ERROR(cudaEventSynchronize(kernel_end));
        CHECK_ERROR(cudaEventElapsedTime(&elapsed_kernel,kernel_start,kernel_end));
        total_time+=elapsed_kernel;
        phi2_conversion<<<grids,blocks>>>(type_con,eta2_out,heta2_out,i);
        cudaDeviceSynchronize();

        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
        swap(heta1,heta1_out);
        swap(heta2,heta2_out);
    }
    CHECK_ERROR(cudaEventDestroy(kernel_start));
    CHECK_ERROR(cudaEventDestroy(kernel_end));
    ofstream ftime("time_tmp.csv");
    ftime<<total_time;
    ftime.close();
    #endif

    #ifdef GET_RESULT
    string s="./result/";
        if(access(s.c_str(),0777)==-1){
            mkdir(s.c_str(),0777);
        }

        #ifdef End2end
            s=s+"end2end/";
            if(access(s.c_str(),0777)==-1){
                mkdir(s.c_str(),0777);
            }
            s=s+string(argv[2])+"/";
            if(access(s.c_str(),0777)==-1){
                mkdir(s.c_str(),0777);
            }
            #ifdef PURE
                s=s+"pure/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                s=s+"para"+string(argv[1])+"/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                writetocsv(s+to_string(timesteps)+".csv",con,dimX,dimY); 
            #endif
            #ifdef AMSTENCIL
                data_sychro_aftercomputation<<<grids_dataprepare,blocks_dataprepare>>>(con,hcon,type_con);
                cudaDeviceSynchronize();
                s=s+"amstencil/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                #ifdef Monitor1
                s=s+"monitor1/";
                string thresholdstr = to_string(threshold);
                #endif
                #ifdef Monitor2
                s=s+"monitor2/";
                ostringstream o_tmp;
                o_tmp<<threshold;
                istringstream i_tmp(o_tmp.str());
                string thresholdstr;
                i_tmp>>thresholdstr;
                #endif
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                s=s+"para"+string(argv[1])+"/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                writetocsv(s+to_string(timesteps)+"_"+thresholdstr+".csv",con,dimX,dimY);
            #endif
            
            #ifdef GRAM1
                data_sychro_aftercomputation<<<grids_dataprepare,blocks_dataprepare>>>(con,hcon,type_con);
                cudaDeviceSynchronize();
                s=s+"gram1/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                s=s+"para"+string(argv[1])+"/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                writetocsv(s+to_string(timesteps)+"_"+to_string(ratio)+".csv",con,dimX,dimY);
            #endif

            #ifdef GRAM2
                data_sychro_aftercomputation<<<grids_dataprepare,blocks_dataprepare>>>(con,hcon,type_con);
                cudaDeviceSynchronize();
                s=s+"gram2/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                s=s+"para"+string(argv[1])+"/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                writetocsv(s+to_string(timesteps)+"_"+to_string(ratio)+".csv",con,dimX,dimY);
            #endif
        #endif
        #ifdef Motivation
            s=s+"motivation/";
            if(access(s.c_str(),0777)==-1){
                mkdir(s.c_str(),0777);
            }
            s=s+string(argv[2])+"/";
            if(access(s.c_str(),0777)==-1){
                mkdir(s.c_str(),0777);
            }
            #ifdef PURE
                s=s+"pure/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                s=s+"para"+string(argv[1])+"/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                writetocsv(s+to_string(timesteps)+".csv",con,dimX,dimY); 
            #endif
            #ifdef AMSTENCIL
                data_sychro_aftercomputation<<<grids_dataprepare,blocks_dataprepare>>>(con,hcon,type_con);
                cudaDeviceSynchronize();
                s=s+"amstencil/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                #ifdef Monitor1
                s=s+"monitor1/";
                #endif
                #ifdef Monitor2
                s=s+"monitor2/";
                #endif
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                s=s+"para"+string(argv[1])+"/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                writetocsv(s+to_string(timesteps)+"_"+to_string(ratio)+".csv",con,dimX,dimY);
            #endif
        
            #ifdef GRAM1
                data_sychro_aftercomputation<<<grids_dataprepare,blocks_dataprepare>>>(con,hcon,type_con);
                cudaDeviceSynchronize();
                s=s+"gram1/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                s=s+"para"+string(argv[1])+"/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                writetocsv(s+to_string(timesteps)+"_"+to_string(ratio)+".csv",con,dimX,dimY);
            #endif

            #ifdef GRAM2
                data_sychro_aftercomputation<<<grids_dataprepare,blocks_dataprepare>>>(con,hcon,type_con);
                cudaDeviceSynchronize();
                s=s+"gram2/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                s=s+"para"+string(argv[1])+"/";
                if(access(s.c_str(),0777)==-1){
                    mkdir(s.c_str(),0777);
                }
                writetocsv(s+to_string(timesteps)+"_"+to_string(ratio)+".csv",con,dimX,dimY);
            #endif
        #endif
    #endif
    CHECK_ERROR(cudaFree(con));
    CHECK_ERROR(cudaFree(eta1));
    CHECK_ERROR(cudaFree(eta2));
    CHECK_ERROR(cudaFree(eta1_out));
    CHECK_ERROR(cudaFree(eta2_out));
    CHECK_ERROR(cudaFree(dummy));
    CHECK_ERROR(cudaFree(con_lap));
    CHECK_ERROR(cudaFree(eta1_lap));
    CHECK_ERROR(cudaFree(eta2_lap));
    CHECK_ERROR(cudaFree(dummy_lap));
    CHECK_ERROR(cudaFree(dfdcon));
    CHECK_ERROR(cudaFree(dfdeta1));
    CHECK_ERROR(cudaFree(dfdeta2));
    #if((defined AMSTENCIL)||(defined GRAM1)||(defined GRAM2))
    CHECK_ERROR(cudaFree(hcon));
    CHECK_ERROR(cudaFree(heta1));
    CHECK_ERROR(cudaFree(heta2));
    CHECK_ERROR(cudaFree(heta1_out));
    CHECK_ERROR(cudaFree(heta2_out));
    CHECK_ERROR(cudaFree(hdummy));
    CHECK_ERROR(cudaFree(hcon_lap));
    CHECK_ERROR(cudaFree(heta1_lap));
    CHECK_ERROR(cudaFree(heta2_lap));
    CHECK_ERROR(cudaFree(hdummy_lap));
    CHECK_ERROR(cudaFree(hdfdcon));
    CHECK_ERROR(cudaFree(hdfdeta1));
    CHECK_ERROR(cudaFree(hdfdeta2));
    #endif
    #ifdef AMSTENCIL
    CHECK_ERROR(cudaFree(type_con));
    CHECK_ERROR(cudaFree(type_old));
    CHECK_ERROR(cudaFree(max_diff_con));
    #endif
    return 0;

}