#include "./paras/para5.h"
#include "stdio.h"
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <sstream>
#include <mma.h>
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <math.h>
using namespace std;
using namespace nvcuda;
#define DEBUG_PRINT
// #define DEBUG_PRINT
#define PURE
// define GET_TIME表示要获取程序的执行时间，GET_RESULT表示获取程序的结果,GET_MonitorTIME表示获取amstencil的monitor时间,monitor_conversion_independent表示获取不加monitor和type conversion的时间
#define GET_RESULT
#define Motivation
#define Monitor2
#ifdef AMSTENCIL
    typedef half2 lowprecision;
    const int lowprecison_dimX=dimXd2;
#else  
    typedef half lowprecision;
    const int lowprecison_dimX=dimX;
#endif
#define CHECK_ERROR(error) checkCudaError(error, __FILE__, __LINE__)
#define CHECK_STATE(msg) checkCudaState(msg, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t error, const char *file, const int line)
{
   if (error != cudaSuccess) {
      std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
      exit(EXIT_FAILURE);
   }
}
inline void checkCudaState(const char *msg, const char *file, const int line)
{
   cudaError_t error = cudaGetLastError();
   if (error != cudaSuccess) {
      std::cerr << "---" << msg << " Error---" << std::endl;
      std::cerr << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
      exit(EXIT_FAILURE);
   }
}
#define monitor_conversion_dependent
//浓度场part1
__global__ void con1_pure(highprecision *con,highprecision *con_lap,highprecision *eta1,highprecision *eta2,highprecision *dfdcon,highprecision *dummy,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*unitx;
    int x=x_start+x_offset;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    highprecision(*con_lapd)[dimX]=(highprecision(*)[dimX])con_lap;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    highprecision(*dfdcond)[dimX]=(highprecision(*)[dimX])dfdcon;
    highprecision(*dummyd)[dimX]=(highprecision(*)[dimX])dummy;
    con_lapd[y][x]=(cond[y][xs1]+cond[y][xa1]+cond[ys1][x]+cond[ya1][x]-4.0*cond[y][x])/dxdy;
    //对浓度场求导
    highprecision sum2=pow(eta1d[y][x],2)+pow(eta2d[y][x],2);
    highprecision sum3=pow(eta1d[y][x],3)+pow(eta2d[y][x],3);
    dfdcond[y][x]=1.0*(2.0*cond[y][x]+4.0*sum3-6.0*sum2)-2.0*16.0*pow(cond[y][x],2)*(1.0-cond[y][x])+2.0*16.0*cond[y][x]*pow(1.0-cond[y][x],2);
    dummyd[y][x]=dfdcond[y][x]-0.5*coefm*con_lapd[y][x];
    // #ifdef DEBUG_PRINT
    // if(print==1 and i>=testtimestep1 and i<=testtimestep2 and testx==x and testy==y){
    //     printf("%d---ys1:%e,ya1:%e,xs1:%e,xa1:%e,center:%e\n",i,cond[ys1][x],cond[ya1][x],cond[y][xs1],cond[y][xa1],cond[y][x]);
    //     printf("%d---dummy:%e,dfdcon:%e,conlap:%e\n",i,dummyd[y][x],dfdcond[y][x],con_lapd[y][x]);
    // }
    // #endif
}
__global__ void con2_pure(highprecision* dummy,highprecision* dummy_lap,highprecision* con,highprecision* eta1,highprecision* eta2,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*unitx;
    int x=x_start+x_offset;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    highprecision(*dummyd)[dimX]=(highprecision(*)[dimX])dummy;
    highprecision(*dummy_lapd)[dimX]=(highprecision(*)[dimX])dummy_lap;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    dummy_lapd[y][x]=(dummyd[y][xs1]+dummyd[y][xa1]+dummyd[ys1][x]+dummyd[ya1][x]-4.0*dummyd[y][x])/dxdy;
    highprecision phi=pow(cond[y][x],3)*(10.0-15.0*cond[y][x]+6.0*pow(cond[y][x],2)); //插值函数
    highprecision sum=eta1d[y][x]*eta2d[y][x]*2;
    highprecision mobil=dvol*phi+dvap*(1.0-phi)+dsur*cond[y][x]*(1.0-cond[y][x])+dgrb*sum;
    cond[y][x]=cond[y][x]+dtime*mobil*dummy_lapd[y][x];
    if(cond[y][x]>=1) cond[y][x]=1;
    else if(cond[y][x]<0)cond[y][x]=0;
    // #ifdef DEBUG_PRINT
    // if(print1 and i>=testtimestep1 and i<=testtimestep2 and x==testx and y==testy){
    //     printf("mixhigh: %d----con:%e,eta1:%e,eta2:%e,dummylap:%e\n",i,cond[y][x],eta1d[y][x],eta2d[y][x],dummy_lapd[y][x]);
    //     printf("mixhigh: %d----dummy:%e,xs1:%e,xa1:%e,ys1:%e,ya1:%e\n",i,dummyd[y][x],dummyd[y][xs1],dummyd[y][xa1],dummyd[ys1][x],dummyd[ya1][x]);
    // }
    // #endif
}
//相场变化
__global__ void phi1_pure(highprecision* eta1,highprecision* eta1_out,highprecision* eta2,highprecision* eta1_lap,highprecision* dfdeta1,highprecision* con,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*unitx;
    int x=x_start+x_offset;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    highprecision(*eta1_outd)[dimX]=(highprecision(*)[dimX])eta1_out;
    highprecision(*eta1_lapd)[dimX]=(highprecision(*)[dimX])eta1_lap;
    highprecision(*dfdeta1d)[dimX]=(highprecision(*)[dimX])dfdeta1;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    //lap算子
    eta1_lapd[y][x]=(eta1d[y][xs1]+eta1d[y][xa1]+eta1d[ys1][x]+eta1d[ya1][x]-4.0*eta1d[y][x])/dxdy;
    //自由能对相求导
    highprecision sum2=pow(eta1d[y][x],2)+pow(eta2d[y][x],2);
    dfdeta1d[y][x]=1.0*(-12.0*pow(eta1d[y][x],2)*(2.0-cond[y][x])+12.0*eta1d[y][x]*(1.0-cond[y][x])+12.0*eta1d[y][x]*sum2);
    eta1_outd[y][x]=eta1d[y][x]-dtime*coefl*(dfdeta1d[y][x]-0.5*coefk*eta1_lapd[y][x]);
    if(eta1_outd[y][x]>=1) eta1_outd[y][x]=1;
    else if(eta1_outd[y][x]<0)eta1_outd[y][x]=0;
}
__global__ void phi2_pure(highprecision* eta2,highprecision* eta2_out,highprecision* eta1,highprecision* eta2_lap,highprecision* dfdeta2,highprecision* con,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    highprecision(*eta2_outd)[dimX]=(highprecision(*)[dimX])eta2_out;
    highprecision(*eta2_lapd)[dimX]=(highprecision(*)[dimX])eta2_lap;
    highprecision(*dfdeta2d)[dimX]=(highprecision(*)[dimX])dfdeta2;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*unitx;
    int x=x_start+x_offset;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    //lap算子
    eta2_lapd[y][x]=(eta2d[y][xs1]+eta2d[y][xa1]+eta2d[ys1][x]+eta2d[ya1][x]-4.0*eta2d[y][x])/dxdy;
    //自由能对相求导
    highprecision sum2=pow(eta1d[y][x],2)+pow(eta2d[y][x],2);
    dfdeta2d[y][x]=1.0*(-12.0*pow(eta2d[y][x],2)*(2.0-cond[y][x])+12.0*eta2d[y][x]*(1.0-cond[y][x])+12.0*eta2d[y][x]*sum2);
    eta2_outd[y][x]=eta2d[y][x]-dtime*coefl*(dfdeta2d[y][x]-0.5*coefk*eta2_lapd[y][x]);
    if(eta2_outd[y][x]>=1) eta2_outd[y][x]=1;
    else if(eta2_outd[y][x]<0)eta2_outd[y][x]=0;
}
#if ((defined AMSTENCIL)||(defined GRAM1||(defined GRAM2)))
__global__ void con1_mix(int *type_curr,highprecision *con,highprecision *con_last,highprecision *con_lap,highprecision *eta1,highprecision *eta2,highprecision *dfdcon,highprecision *dummy,lowprecision *hcon,lowprecision *hcon_lap,lowprecision *heta1,lowprecision *heta2,lowprecision *hdfdcon,lowprecision* hdummy,lowprecision hcf,lowprecision hdxdy,lowprecision hsix,lowprecision hzpfive,lowprecision hone,lowprecision htwo,lowprecision hfour,lowprecision hthirtytwo,lowprecision hcoefm,int i){

    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;

    int unitindex_ys1=unitindex_y==0?unitdimY-1:unitindex_y-1;
    int unitindex_ya1=unitindex_y==unitdimY-1?0:unitindex_y+1;
    int type;
    #ifdef AMSTENCIL
        int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
        type=type_currd[unitindex_y][unitindex_x];
    #endif
    #ifdef GRAM1
        if(blockIdx.z<unitNums*ratio)type=1;
        else type=2;
    #endif
    #ifdef GRAM2
        if(blockIdx.z%100<100*ratio)type=1;
        else type=2;
    #endif
    int x_offset=threadIdx.x;
    int x_start=unitindex_x*uxd2;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    int ys1=y>0?y-1:dimY-1;
    int ya1=y<dimY-1?y+1:0;
    lowprecision(*hdfdcond)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hdfdcon;
    lowprecision(*hdummyd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hdummy;
    lowprecision(*hcon_lapd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hcon_lap;
    lowprecision(*hcond)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hcon;
    lowprecision(*heta1d)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta1;
    lowprecision(*heta2d)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta2;
    highprecision(*dfdcond)[dimX]=(highprecision(*)[dimX])dfdcon;
    highprecision(*dummyd)[dimX]=(highprecision(*)[dimX])dummy;
    highprecision(*con_lapd)[dimX]=(highprecision(*)[dimX])con_lap;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    #ifdef monitor_conversion_dependent
    #ifdef Monitor2
    if(i%10==0){
        highprecision(*con_lastd)[dimX]=(highprecision(*)[dimX])con_last;
        con_lastd[y][x_start*2+x_offset+uxd2*blockIdx.x]=cond[y][x_start*2+x_offset+uxd2*blockIdx.x];
    }
    #endif
    #endif
    if(type==1){
        //计算点的坐标
        #ifdef AMSTENCIL
        if(blockIdx.x==1)return;
        int x=x_start+x_offset;
        #else
        x_offset=x_offset+blockIdx.x*uxd2;
        int x=x_start*2+x_offset;
        #endif
        int xs1=x>0?x-1:lowprecison_dimX-1;
        int xa1=x<lowprecison_dimX-1?x+1:0;
        
        // 计算数据准备，amstencil与gram1,gram2比较邻居获取方式不同
        #ifdef AMSTENCIL    
        lowprecision con_xs1=x_offset>0?hcond[y][xs1]:__halves2half2(__high2half(hcond[y][xs1]),__low2half(hcond[y][x+uxd2-1]));
        lowprecision con_xa1=x_offset<uxd2-1?hcond[y][xa1]:__halves2half2(__high2half(hcond[y][x-uxd2+1]),__low2half(hcond[y][xa1]));
        lowprecision con_ya1=(y_offset==uys1&&type_currd[unitindex_ya1][unitindex_x]==2)?__floats2half2_rn(cond[ya1][x_start*2+x_offset],cond[ya1][x_start*2+x_offset+uxd2]):hcond[ya1][x];
        lowprecision con_ys1=(y_offset==0&&type_currd[unitindex_ys1][unitindex_x]==2)?__floats2half2_rn(cond[ys1][x_start*2+x_offset],cond[ys1][x_start*2+x_offset+uxd2]):hcond[ys1][x];
        #endif

        #ifdef GRAM1
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        lowprecision con_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?hcond[y][xs1]:__float2half(cond[y][xs1]);
        lowprecision con_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?hcond[y][xa1]:__float2half(cond[y][xa1]);
        lowprecision con_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?hcond[ys1][x]:__float2half(cond[ys1][x]);
        lowprecision con_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?hcond[ya1][x]:__float2half(cond[ya1][x]);
        #endif

        #ifdef GRAM2
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        lowprecision con_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?hcond[y][xs1]:__float2half(cond[y][xs1]);
        lowprecision con_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?hcond[y][xa1]:__float2half(cond[y][xa1]);
        lowprecision con_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?hcond[ys1][x]:__float2half(cond[ys1][x]);
        lowprecision con_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?hcond[ya1][x]:__float2half(cond[ya1][x]);      
        #endif
        // 计算
        hcon_lapd[y][x]=(con_xs1+con_xa1+con_ys1+con_ya1-hcf*hcond[y][x])/hdxdy;
        lowprecision sum2=heta1d[y][x]*heta1d[y][x]+heta2d[y][x]*heta2d[y][x];
        lowprecision sum3=heta1d[y][x]*heta1d[y][x]*heta1d[y][x]+heta2d[y][x]*heta2d[y][x]*heta2d[y][x];
        hdfdcond[y][x]=(htwo*hcond[y][x]+hfour*sum3-hsix*sum2)-hthirtytwo*(hcond[y][x]*hcond[y][x])*(hone-hcond[y][x])+hthirtytwo*hcond[y][x]*((hone-hcond[y][x])*(hone-hcond[y][x]));
        hdummyd[y][x]=hdfdcond[y][x]-hzpfive*hcoefm*hcon_lapd[y][x];
        // 数据写回，只有amstencil需要写回数据
        #ifdef monitor_conversion_dependent
        #ifdef AMSTENCIL
        if(y_offset==0||y_offset==uys1){
            dummyd[y][x_start*2+x_offset]=__low2float(hdummyd[y][x]);
            dummyd[y][x_start*2+x_offset+uxd2]=__high2float(hdummyd[y][x]);
        }
        else if(x_offset==0)dummyd[y][x_start*2]=__low2float(hdummyd[y][x]);
        else if(x_offset==uxd2s1)dummyd[y][x_start*2+x_offset+uxd2]=__high2float(hdummyd[y][x]);
        // 改bug时打印一些数据
        #endif
        #endif
        // #ifdef DEBUG_PRINT
        // if(print==1 and i>=testtimestep1 and i<=testtimestep2 and htestx==x and testy==y){
        //     if(islowpart){
        //         printf("%d---half low ys1:%e,ya1:%e,xs1:%e,xa1:%e,center:%e\n",i,__low2float(con_ys1),__low2float(con_ya1),__low2float(con_xs1),__low2float(con_xa1),__low2float(hcond[y][x]));
        //         printf("%d---half low dummy:%e,dfdcon:%e,conlap:%e\n",i,__low2float(hdummyd[y][x]),__low2float(hdfdcond[y][x]),__low2float(hcon_lapd[y][x]));
        //     }
        //     else{
        //         printf("%d---half high ys1:%e,ya1:%e,xs1:%e,xa1:%e,center:%e\n",i,__high2float(con_ys1),__high2float(con_ya1),__high2float(con_xs1),__high2float(con_xa1),__high2float(hcond[y][x]));
        //         printf("%d---half high dummy:%e,dfdcon:%e,conlap:%e\n",i,__high2float(hdummyd[y][x]),__high2float(hdfdcond[y][x]),__high2float(hcon_lapd[y][x]));
        //     }
        // }
        // #endif 

    }
    //double
    else{

        x_offset=x_offset+blockIdx.x*uxd2;
        int x=x_start*2+x_offset;
        int xs1=x>0?x-1:dimX-1;
        int xa1=x<dimX-1?x+1:0;
        #ifdef AMSTENCIL
        con_lapd[y][x]=(cond[y][xs1]+cond[y][xa1]+cond[ys1][x]+cond[ya1][x]-4.0*cond[y][x])/dxdy;
        #endif
        #ifdef GRAM1
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        highprecision con_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?cond[y][xs1]:__half2float(hcond[y][xs1]);
        highprecision con_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?cond[y][xa1]:__half2float(hcond[y][xa1]);
        highprecision con_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?cond[ys1][x]:__half2float(hcond[ys1][x]);
        highprecision con_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?cond[ya1][x]:__half2float(hcond[ya1][x]);
        con_lapd[y][x]=(con_xs1+con_xa1+con_ys1+con_ya1-4.0*cond[y][x])/dxdy; 
        #endif
        #ifdef GRAM2
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        highprecision con_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?cond[y][xs1]:__half2float(hcond[y][xs1]);
        highprecision con_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?cond[y][xa1]:__half2float(hcond[y][xa1]);
        highprecision con_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?cond[ys1][x]:__half2float(hcond[ys1][x]);
        highprecision con_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?cond[ya1][x]:__half2float(hcond[ya1][x]);
        con_lapd[y][x]=(con_xs1+con_xa1+con_ys1+con_ya1-4.0*cond[y][x])/dxdy;
        #endif

        highprecision sum2=pow(eta1d[y][x],2)+pow(eta2d[y][x],2);
        highprecision sum3=pow(eta1d[y][x],3)+pow(eta2d[y][x],3);
        dfdcond[y][x]=1.0*(2.0*cond[y][x]+4.0*sum3-6.0*sum2)-2.0*16.0*pow(cond[y][x],2)*(1.0-cond[y][x])+2.0*16.0*cond[y][x]*pow(1.0-cond[y][x],2);
        dummyd[y][x]=dfdcond[y][x]-0.5*coefm*con_lapd[y][x];
        // 数据写回
        #ifdef monitor_conversion_dependent
        #ifdef AMSTENCIL
        if(x_offset==0)hdummyd[y][x_start]=__halves2half2(__float2half(dummyd[y][x]),__high2half(hdummyd[y][x_start]));
        if(x_offset==uxs1)hdummyd[y][x_start+x_offset-uxd2]=__halves2half2(__low2half(hdummyd[y][x_start+x_offset-uxd2]),__float2half(dummyd[y][x]));
        #endif
        #endif
        // #ifdef DEBUG_PRINT
        //     #ifdef AMSTENCIL
        //         if(i==1500 and x==255 and y==272){
        //             printf("lap:%e, ys1:%e,ya1:%e,xs1:%e,xa1:%e,center:%e\n",con_lapd[y][x],cond[ys1][x],cond[ya1][x],cond[y][xs1],cond[y][xa1],cond[y][x]);
        //             printf(" dummy:%e,dfdcon:%e,conlap:%e\n",dummyd[y][x],dfdcond[y][x],con_lapd[y][x]);
        //         }
        //     #else
        //         if(i==1500 and x==255 and y==272){
        //             printf("lap:%e, ys1:%e,ya1:%e,xs1:%e,xa1:%e,center:%e\n",con_lapd[y][x],con_ys1,con_ya1,con_xs1,con_xa1,cond[y][x]);
        //             printf(" dummy:%e,dfdcon:%e,conlap:%e\n",dummyd[y][x],dfdcond[y][x],con_lapd[y][x]);
        //         }
        //     #endif
        // #endif
    }
}
__global__ void con2_mix(int *type_curr,highprecision *dummy,highprecision *dummy_lap,highprecision *con,highprecision *eta1,highprecision *eta2,lowprecision *hdummy,lowprecision *hdummy_lap,lowprecision *hcon,lowprecision *heta1,lowprecision *heta2,lowprecision hcf,lowprecision hdxdy,lowprecision hone,lowprecision htwo,lowprecision hsix,lowprecision hten,lowprecision hfifteen,lowprecision hdgrb,lowprecision hdsur,lowprecision hdtime,lowprecision hdvap,lowprecision hdvol,lowprecision highbound,lowprecision lowbound,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int unitindex_ys1=unitindex_y==0?unitdimY-1:unitindex_y-1;
    int unitindex_ya1=unitindex_y==unitdimY-1?0:unitindex_y+1;
    int type;
    #ifdef AMSTENCIL
        int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
        type=type_currd[unitindex_y][unitindex_x];
    #endif
    #ifdef GRAM1
        if(blockIdx.z<unitNums*ratio)type=1;
        else type=2;
    #endif
    #ifdef GRAM2
        if(blockIdx.z%100<100*ratio)type=1;
        else type=2;
    #endif
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*uxd2;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    int ys1=y>0?y-1:dimY-1;int ya1=y<dimY-1?y+1:0;
    highprecision(*dummy_lapd)[dimX]=(highprecision(*)[dimX])dummy_lap;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    highprecision(*dummyd)[dimX]=(highprecision(*)[dimX])dummy;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    lowprecision(*hdummy_lapd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hdummy_lap;
    lowprecision(*hcond)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hcon;
    lowprecision(*hdummyd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hdummy;
    lowprecision(*heta1d)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta1;
    lowprecision(*heta2d)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta2;
    //half2
    if(type==1){
        #ifdef AMSTENCIL
        if(blockIdx.x==1)return;
        int x=x_start+x_offset;
        #else
        x_offset=x_offset+blockIdx.x*uxd2;
        int x=x_start*2+x_offset;
        #endif
        int xs1=x>0?x-1:lowprecison_dimX-1;
        int xa1=x<lowprecison_dimX-1?x+1:0;
        // 计算数据准备，amstencil与gram1,gram2比较邻居获取方式不同
        #ifdef AMSTENCIL    
        lowprecision dummy_xs1=x_offset>0?hdummyd[y][xs1]:__halves2half2(__high2half(hdummyd[y][xs1]),__low2half(hdummyd[y][x+uxd2-1]));
        lowprecision dummy_xa1=x_offset<uxd2-1?hdummyd[y][xa1]:__halves2half2(__high2half(hdummyd[y][x-uxd2+1]),__low2half(hdummyd[y][xa1]));
        lowprecision dummy_ys1=(y_offset==0&&type_currd[unitindex_ys1][unitindex_x]==2)?__floats2half2_rn(dummyd[ys1][x_start*2+x_offset],dummyd[ys1][x_start*2+x_offset+uxd2]):hdummyd[ys1][x];
        lowprecision dummy_ya1=(y_offset==uys1&&type_currd[unitindex_ya1][unitindex_x]==2)?__floats2half2_rn(dummyd[ya1][x_start*2+x_offset],dummyd[ya1][x_start*2+x_offset+uxd2]):hdummyd[ya1][x];
        #endif

        #ifdef GRAM1
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        lowprecision dummy_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?hdummyd[y][xs1]:__float2half(dummyd[y][xs1]);
        lowprecision dummy_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?hdummyd[y][xa1]:__float2half(dummyd[y][xa1]);
        lowprecision dummy_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?hdummyd[ys1][x]:__float2half(dummyd[ys1][x]);
        lowprecision dummy_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?hdummyd[ya1][x]:__float2half(dummyd[ya1][x]);
        #endif

        #ifdef GRAM2
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        lowprecision dummy_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?hdummyd[y][xs1]:__float2half(dummyd[y][xs1]);
        lowprecision dummy_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?hdummyd[y][xa1]:__float2half(dummyd[y][xa1]);
        lowprecision dummy_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?hdummyd[ys1][x]:__float2half(dummyd[ys1][x]);
        lowprecision dummy_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?hdummyd[ya1][x]:__float2half(dummyd[ya1][x]);      
        #endif

        hdummy_lapd[y][x]=(dummy_xs1+dummy_xa1+dummy_ys1+dummy_ya1-hcf*hdummyd[y][x])/hdxdy;
        lowprecision phi=hcond[y][x]*hcond[y][x]*hcond[y][x]*(hten-hfifteen*hcond[y][x]+hsix*(hcond[y][x]*hcond[y][x]));
        lowprecision sum=heta1d[y][x]*heta2d[y][x]*htwo;
        lowprecision mobil=hdvol*phi+hdvap*(hone-phi)+hdsur*hcond[y][x]*(hone-hcond[y][x])+hdgrb*sum;
        hcond[y][x]=hcond[y][x]+hdtime*mobil*hdummy_lapd[y][x];
        #ifdef AMSTENCIL
        hcond[y][x]=__hmin2(highbound,hcond[y][x]);
        hcond[y][x]=__hmax2(lowbound,hcond[y][x]);
        #else
        hcond[y][x]=__hmin(highbound,hcond[y][x]);
        hcond[y][x]=__hmax(lowbound,hcond[y][x]);
        #endif
        #ifdef monitor_conversion_dependent
        #ifdef AMSTENCIL
        if(y_offset==0||y_offset==uys1||(i+1)%10==0){
            cond[y][x_start*2+x_offset]=__low2float(hcond[y][x]);
            cond[y][x_start*2+x_offset+uxd2]=__high2float(hcond[y][x]);
        }
        else if(x_offset==0)cond[y][x_start*2]=__low2float(hcond[y][x]);
        else if(x_offset==uxd2s1)cond[y][x_start*2+x_offset+uxd2]=__high2float(hcond[y][x]); 
        #endif
        #endif
        // #ifdef DEBUG_PRINT
        // if(print1 and i>=testtimestep1 and i<=testtimestep2 and x==htestx and y==testy){
        //     if(islowpart){
        //         printf("mixhalflow: %d----con:%e,eta1:%e,eta2:%e,dummylap:%e\n",i,__low2float(hcond[y][x]),__low2float(heta1d[y][x]),__low2float(heta2d[y][x]),__low2float(hdummy_lapd[y][x]));
        //         printf("mixhalflow: %d----dummy xs1:%e,xa1:%e,ys1:%e,ya1:%e\n",i,__low2float(dummy_xs1),__low2float(dummy_xa1),__low2float(dummy_ys1),__low2float(dummy_ya1));
        //     }
        //     else{
        //         printf("mixhalfhigh: %d----con:%e,eta1:%e,eta2:%e,dummylap:%e\n",i,__high2float(hcond[y][x]),__high2float(heta1d[y][x]),__high2float(heta2d[y][x]),__high2float(hdummy_lapd[y][x]));
        //         printf("mixhalfhigh: %d----dummy xs1:%e,xa1:%e,ys1:%e,ya1:%e\n",i,__high2float(dummy_xs1),__high2float(dummy_xa1),__high2float(dummy_ys1),__high2float(dummy_ya1));
        //     }
        // } 
        // #endif
    }
    //double
    else{
        x_offset=x_offset+uxd2*blockIdx.x;
        int x=x_start*2+x_offset;
        int xs1=x>0?x-1:dimX-1;
        int xa1=x<dimX-1?x+1:0;
        #ifdef AMSTENCIL
        dummy_lapd[y][x]=(dummyd[y][xs1]+dummyd[y][xa1]+dummyd[ys1][x]+dummyd[ya1][x]-4.0*dummyd[y][x])/dxdy;
        #endif
        #ifdef GRAM1
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        highprecision dummy_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?dummyd[y][xs1]:__half2float(hdummyd[y][xs1]);
        highprecision dummy_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?dummyd[y][xa1]:__half2float(hdummyd[y][xa1]);
        highprecision dummy_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?dummyd[ys1][x]:__half2float(hdummyd[ys1][x]);
        highprecision dummy_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?dummyd[ya1][x]:__half2float(hdummyd[ya1][x]);
        dummy_lapd[y][x]=(dummy_xs1+dummy_xa1+dummy_ys1+dummy_ya1-4.0*dummyd[y][x])/dxdy; 
        #endif
        #ifdef GRAM2
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        highprecision dummy_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?dummyd[y][xs1]:__half2float(hdummyd[y][xs1]);
        highprecision dummy_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?dummyd[y][xa1]:__half2float(hdummyd[y][xa1]);
        highprecision dummy_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?dummyd[ys1][x]:__half2float(hdummyd[ys1][x]);
        highprecision dummy_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?dummyd[ya1][x]:__half2float(hdummyd[ya1][x]);
        dummy_lapd[y][x]=(dummy_xs1+dummy_xa1+dummy_ys1+dummy_ya1-4.0*dummyd[y][x])/dxdy;
        #endif
        highprecision phi=pow(cond[y][x],3)*(10.0-15.0*cond[y][x]+6.0*pow(cond[y][x],2)); //插值函数
        highprecision sum=eta1d[y][x]*eta2d[y][x]*2;
        highprecision mobil=dvol*phi+dvap*(1.0-phi)+dsur*cond[y][x]*(1.0-cond[y][x])+dgrb*sum;
        cond[y][x]=cond[y][x]+dtime*mobil*dummy_lapd[y][x];
        if(cond[y][x]>=1) cond[y][x]=1;
        else if(cond[y][x]<0)cond[y][x]=0;
        #ifdef monitor_conversion_dependent
        #ifdef AMSTENCIL
        if(x_offset==0)hcond[y][x_start]=__halves2half2(__float2half(cond[y][x]),__high2half(hcond[y][x_start]));
        if(x_offset==uxs1)hcond[y][x_start+x_offset-uxd2]=__halves2half2(__low2half(hcond[y][x_start+x_offset-uxd2]),__float2half(cond[y][x]));
        #endif
        #endif
        // #ifdef DEBUG_PRINT
        // if(i==1500 and x==255 and y==272){
        //     printf("----con:%e,eta1:%e,eta2:%e,dummylap:%e\n",cond[y][x],eta1d[y][x],eta2d[y][x],dummy_lapd[y][x]);
        //     printf("---dummy:%e,xs1:%e,xa1:%e,ys1:%e,ya1:%e\n",dummyd[y][x],dummyd[y][xs1],dummyd[y][xa1],dummyd[ys1][x],dummyd[ya1][x]);
        // }
        // #endif
    }
}
__global__ void phi1_mix(int* type_curr,highprecision* eta1,highprecision* eta1_out,highprecision* eta2,highprecision* eta1_lap,highprecision* dfdeta1,highprecision* con,lowprecision* heta1,lowprecision* heta1_out,lowprecision* heta2,lowprecision* heta1_lap,lowprecision* hdfdeta1,lowprecision* hcon,lowprecision hcf,lowprecision hdxdy,lowprecision hzpfive,lowprecision hone,lowprecision htwo,lowprecision htwelve,lowprecision hdtime,lowprecision hcoefl,lowprecision hcoefk,lowprecision highbound,lowprecision lowbound,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int unitindex_ys1=unitindex_y==0?unitdimY-1:unitindex_y-1;
    int unitindex_ya1=unitindex_y==unitdimY-1?0:unitindex_y+1;
    int type;
    #ifdef AMSTENCIL
        int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
        type=type_currd[unitindex_y][unitindex_x];
    #endif
    #ifdef GRAM1
        if(blockIdx.z<unitNums*ratio)type=1;
        else type=2;
    #endif
    #ifdef GRAM2
        if(blockIdx.z%100<100*ratio)type=1;
        else type=2;
    #endif
    
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*uxd2;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    highprecision(*eta1_outd)[dimX]=(highprecision(*)[dimX])eta1_out;
    highprecision(*eta1_lapd)[dimX]=(highprecision(*)[dimX])eta1_lap;
    highprecision(*dfdeta1d)[dimX]=(highprecision(*)[dimX])dfdeta1;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    lowprecision(*heta1_outd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta1_out;
    lowprecision(*heta1_lapd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta1_lap;
    lowprecision(*hdfdeta1d)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hdfdeta1;
    lowprecision(*heta1d)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta1;
    lowprecision(*heta2d)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta2;
    lowprecision(*hcond)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hcon;

    int ys1=y>0?y-1:dimY-1;
    int ya1=y<dimY-1?y+1:0;
    if(type==1){
        #ifdef AMSTENCIL
        if(blockIdx.x==1)return;
        int x=x_start+x_offset;
        #else
        x_offset=x_offset+blockIdx.x*uxd2;
        int x=x_start*2+x_offset;
        #endif
        int xs1=x>0?x-1:lowprecison_dimX-1;
        int xa1=x<lowprecison_dimX-1?x+1:0;
        #ifdef AMSTENCIL    
        lowprecision eta1_xs1=x_offset>0?heta1d[y][xs1]:__halves2half2(__high2half(heta1d[y][xs1]),__low2half(heta1d[y][x+uxd2-1]));
        lowprecision eta1_xa1=x_offset<uxd2-1?heta1d[y][xa1]:__halves2half2(__high2half(heta1d[y][x-uxd2+1]),__low2half(heta1d[y][xa1]));
        lowprecision eta1_ys1=(y_offset==0&&type_currd[unitindex_ys1][unitindex_x]==2)?__floats2half2_rn(eta1d[ys1][x_start*2+x_offset],eta1d[ys1][x_start*2+x_offset+uxd2]):heta1d[ys1][x];
        lowprecision eta1_ya1=(y_offset==uys1&&type_currd[unitindex_ya1][unitindex_x]==2)?__floats2half2_rn(eta1d[ya1][x_start*2+x_offset],eta1d[ya1][x_start*2+x_offset+uxd2]):heta1d[ya1][x];
        #endif

        #ifdef GRAM1
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        lowprecision eta1_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?heta1d[y][xs1]:__float2half(eta1d[y][xs1]);
        lowprecision eta1_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?heta1d[y][xa1]:__float2half(eta1d[y][xa1]);
        lowprecision eta1_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?heta1d[ys1][x]:__float2half(eta1d[ys1][x]);
        lowprecision eta1_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?heta1d[ya1][x]:__float2half(eta1d[ya1][x]);
        #endif

        #ifdef GRAM2
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        lowprecision eta1_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?heta1d[y][xs1]:__float2half(eta1d[y][xs1]);
        lowprecision eta1_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?heta1d[y][xa1]:__float2half(eta1d[y][xa1]);
        lowprecision eta1_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?heta1d[ys1][x]:__float2half(eta1d[ys1][x]);
        lowprecision eta1_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?heta1d[ya1][x]:__float2half(eta1d[ya1][x]);      
        #endif

        heta1_lapd[y][x]=(eta1_xs1+eta1_xa1+eta1_ys1+eta1_ya1-hcf*heta1d[y][x])/hdxdy;
        lowprecision sum2=(heta1d[y][x]*heta1d[y][x])+(heta2d[y][x]*heta2d[y][x]);
        hdfdeta1d[y][x]=hone*(-htwelve*(heta1d[y][x]*heta1d[y][x])*(htwo-hcond[y][x])+htwelve*heta1d[y][x]*(hone-hcond[y][x])+htwelve*heta1d[y][x]*sum2);
        heta1_outd[y][x]=heta1d[y][x]-hdtime*hcoefl*(hdfdeta1d[y][x]-hzpfive*hcoefk*heta1_lapd[y][x]);
        #ifdef AMSTENCIL
        heta1_outd[y][x]=__hmin2(highbound,heta1_outd[y][x]);
        heta1_outd[y][x]=__hmax2(lowbound,heta1_outd[y][x]);
        #else
        heta1_outd[y][x]=__hmin(highbound,heta1_outd[y][x]);
        heta1_outd[y][x]=__hmax(lowbound,heta1_outd[y][x]);
        #endif
        #ifdef monitor_conversion_dependent
        #ifdef AMSTENCIL
        if(y_offset==0||y_offset==uys1||(i+1)%10==0){
            eta1_outd[y][x_start*2+x_offset]=__low2float(heta1_outd[y][x]);
            eta1_outd[y][x_start*2+x_offset+uxd2]=__high2float(heta1_outd[y][x]);
        }
        else if(x_offset==0)eta1_outd[y][x_start*2]=__low2float(heta1_outd[y][x]);
        else if(x_offset==uxd2s1)eta1_outd[y][x_start*2+x_offset+uxd2]=__high2float(heta1_outd[y][x]);
        #endif
        #endif
        #ifdef DEGUB_PRINT
        if(print2 and i>=testtimestep1 and i<=testtimestep2 and x==htestx and y==testy){
            if(islowpart){
                printf("mixhalflow: %d----eta1out:%e,dfdeta1:%e,eta1lap:%e\n",i,__low2float(heta1_outd[y][x]),__low2float(hdfdeta1d[y][x]),__low2float(heta1_lapd[y][x]));

            }
            else{
                printf("mixhalflow: %d----eta1out:%e,dfdeta1:%e,eta1lap:%e\n",i,__high2float(heta1_outd[y][x]),__high2float(hdfdeta1d[y][x]),__high2float(heta1_lapd[y][x]));
            }
        }
        #endif
    }
    else{
        x_offset=x_offset+uxd2*blockIdx.x;
        int x=x_start*2+x_offset;
        int xs1=x>0?x-1:dimX-1;
        int xa1=x<dimX-1?x+1:0;
        #ifdef AMSTENCIL
        eta1_lapd[y][x]=(eta1d[y][xs1]+eta1d[y][xa1]+eta1d[ys1][x]+eta1d[ya1][x]-4.0*eta1d[y][x])/dxdy;
        #endif
        #ifdef GRAM1
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        highprecision eta1_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?eta1d[y][xs1]:__half2float(heta1d[y][xs1]);
        highprecision eta1_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?eta1d[y][xa1]:__half2float(heta1d[y][xa1]);
        highprecision eta1_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?eta1d[ys1][x]:__half2float(heta1d[ys1][x]);
        highprecision eta1_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?eta1d[ya1][x]:__half2float(heta1d[ya1][x]);
        eta1_lapd[y][x]=(eta1_xs1+eta1_xa1+eta1_ys1+eta1_ya1-4.0*eta1d[y][x])/dxdy; 
        #endif
        #ifdef GRAM2
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        highprecision eta1_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?eta1d[y][xs1]:__half2float(heta1d[y][xs1]);
        highprecision eta1_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?eta1d[y][xa1]:__half2float(heta1d[y][xa1]);
        highprecision eta1_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?eta1d[ys1][x]:__half2float(heta1d[ys1][x]);
        highprecision eta1_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?eta1d[ya1][x]:__half2float(heta1d[ya1][x]);
        eta1_lapd[y][x]=(eta1_xs1+eta1_xa1+eta1_ys1+eta1_ya1-4.0*eta1d[y][x])/dxdy;
        #endif
        //自由能对相求导
        highprecision sum2=pow(eta1d[y][x],2)+pow(eta2d[y][x],2);
        dfdeta1d[y][x]=1.0*(-12.0*pow(eta1d[y][x],2)*(2.0-cond[y][x])+12.0*eta1d[y][x]*(1.0-cond[y][x])+12.0*eta1d[y][x]*sum2);
        eta1_outd[y][x]=eta1d[y][x]-dtime*coefl*(dfdeta1d[y][x]-0.5*coefk*eta1_lapd[y][x]);
        if(eta1_outd[y][x]>=1) eta1_outd[y][x]=1;
        else if(eta1_outd[y][x]<0)eta1_outd[y][x]=0;
        #ifdef monitor_conversion_dependent
        #ifdef AMSTENCIL
        if(x_offset==0)heta1_outd[y][x_start]=__halves2half2(__float2half(eta1_outd[y][x]),__high2half(heta1_outd[y][x_start]));
        if(x_offset==uxs1)heta1_outd[y][x_start+x_offset-uxd2]=__halves2half2(__low2half(heta1_outd[y][x_start+x_offset-uxd2]),__float2half(eta1_outd[y][x]));
        #endif
        #endif
    }
}
__global__ void phi2_mix(int* type_curr,highprecision* eta2,highprecision* eta2_out,highprecision* eta1,highprecision* eta2_lap,highprecision* dfdeta2,highprecision* con,lowprecision* heta2,lowprecision* heta2_out,lowprecision* heta1,lowprecision* heta2_lap,lowprecision* hdfdeta2,lowprecision* hcon,lowprecision hcf,lowprecision hdxdy,lowprecision hzpfive,lowprecision hone,lowprecision htwo,lowprecision htwelve,lowprecision hdtime,lowprecision hcoefl,lowprecision hcoefk,lowprecision highbound,lowprecision lowbound,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int unitindex_ys1=unitindex_y==0?unitdimY-1:unitindex_y-1;
    int unitindex_ya1=unitindex_y==unitdimY-1?0:unitindex_y+1;
    int type;
    #ifdef AMSTENCIL
        int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
        type=type_currd[unitindex_y][unitindex_x];
    #endif
    #ifdef GRAM1
        if(blockIdx.z<unitNums*ratio)type=1;
        else type=2;
    #endif
    #ifdef GRAM2
        if(blockIdx.z%100<100*ratio)type=1;
        else type=2;
    #endif
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*uxd2;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    lowprecision(*heta2_outd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta2_out;
    lowprecision(*heta2_lapd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta2_lap;
    lowprecision(*hdfdeta2d)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hdfdeta2;
    lowprecision(*heta1d)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta1;
    lowprecision(*heta2d)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta2;
    lowprecision(*hcond)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hcon;
    highprecision(*eta2_outd)[dimX]=(highprecision(*)[dimX])eta2_out;
    highprecision(*eta2_lapd)[dimX]=(highprecision(*)[dimX])eta2_lap;
    highprecision(*dfdeta2d)[dimX]=(highprecision(*)[dimX])dfdeta2;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;

    int ys1=y>0?y-1:dimY-1;
    int ya1=y<dimY-1?y+1:0;
    if(type==1){
        #ifdef AMSTENCIL
        if(blockIdx.x==1)return;
        int x=x_start+x_offset;
        #else
        x_offset=x_offset+blockIdx.x*uxd2;
        int x=x_start*2+x_offset;
        #endif
        int xs1=x>0?x-1:lowprecison_dimX-1;
        int xa1=x<lowprecison_dimX-1?x+1:0;
        #ifdef AMSTENCIL    
        lowprecision eta2_xs1=x_offset>0?heta2d[y][xs1]:__halves2half2(__high2half(heta2d[y][xs1]),__low2half(heta2d[y][x+uxd2-1]));
        lowprecision eta2_xa1=x_offset<uxd2-1?heta2d[y][xa1]:__halves2half2(__high2half(heta2d[y][x-uxd2+1]),__low2half(heta2d[y][xa1]));
        lowprecision eta2_ys1=(y_offset==0&&type_currd[unitindex_ys1][unitindex_x]==2)?__floats2half2_rn(eta2d[ys1][x_start*2+x_offset],eta2d[ys1][x_start*2+x_offset+uxd2]):heta2d[ys1][x];
        lowprecision eta2_ya1=(y_offset==uys1&&type_currd[unitindex_ya1][unitindex_x]==2)?__floats2half2_rn(eta2d[ya1][x_start*2+x_offset],eta2d[ya1][x_start*2+x_offset+uxd2]):heta2d[ya1][x];
        #endif

        #ifdef GRAM1
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        lowprecision eta2_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?heta2d[y][xs1]:__float2half(eta2d[y][xs1]);
        lowprecision eta2_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?heta2d[y][xa1]:__float2half(eta2d[y][xa1]);
        lowprecision eta2_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?heta2d[ys1][x]:__float2half(eta2d[ys1][x]);
        lowprecision eta2_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?heta2d[ya1][x]:__float2half(eta2d[ya1][x]);
        #endif
        #ifdef GRAM2
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        lowprecision eta2_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?heta2d[y][xs1]:__float2half(eta2d[y][xs1]);
        lowprecision eta2_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?heta2d[y][xa1]:__float2half(eta2d[y][xa1]);
        lowprecision eta2_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?heta2d[ys1][x]:__float2half(eta2d[ys1][x]);
        lowprecision eta2_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?heta2d[ya1][x]:__float2half(eta2d[ya1][x]);      
        #endif
        heta2_lapd[y][x]=(eta2_xs1+eta2_xa1+eta2_ys1+eta2_ya1-hcf*heta2d[y][x])/hdxdy;
    
        lowprecision sum2=(heta1d[y][x]*heta1d[y][x])+(heta2d[y][x]*heta2d[y][x]);
        hdfdeta2d[y][x]=hone*(-htwelve*(heta2d[y][x]*heta2d[y][x])*(htwo-hcond[y][x])+htwelve*heta2d[y][x]*(hone-hcond[y][x])+htwelve*heta2d[y][x]*sum2);
        heta2_outd[y][x]=heta2d[y][x]-hdtime*hcoefl*(hdfdeta2d[y][x]-hzpfive*hcoefk*heta2_lapd[y][x]);
        #ifdef AMSTENCIL
        heta2_outd[y][x]=__hmin2(highbound,heta2_outd[y][x]);
        heta2_outd[y][x]=__hmax2(lowbound,heta2_outd[y][x]);
        #else
        heta2_outd[y][x]=__hmin(highbound,heta2_outd[y][x]);
        heta2_outd[y][x]=__hmax(lowbound,heta2_outd[y][x]);
        #endif
        #ifdef monitor_conversion_dependent
        #ifdef AMSTENCIL
        if(y_offset==0||y_offset==uys1||(i+1)%10==0){
            eta2_outd[y][x_start*2+x_offset]=__low2float(heta2_outd[y][x]);
            eta2_outd[y][x_start*2+x_offset+uxd2]=__high2float(heta2_outd[y][x]);
        }
        else if(x_offset==0)eta2_outd[y][x_start*2+x_offset]=__low2float(heta2_outd[y][x]);
        else if(x_offset==uxd2s1)eta2_outd[y][x_start*2+x_offset+uxd2]=__high2float(heta2_outd[y][x]);
        #endif
        #endif
        // #ifdef DEBUG_PRINT
        // if(print3 and i>=testtimestep1 and i<=testtimestep2 and x==htestx and y==testy){
        //     if(islowpart){
        //         printf("mixhalflow: %d----eta2out:%e,dfdeta2:%e,eta2lap:%e\n",i,__low2float(heta2_outd[y][x]),__low2float(hdfdeta2d[y][x]),__low2float(heta2_lapd[y][x]));
        //         printf("mixhalflow: %d----eta2:%e, xs1:%e,xa1:%e,ys1:%e,ya1:%e\n",i,__low2float(heta2d[y][x]),__low2float(eta2_xs1),__low2float(eta2_xa1),__low2float(eta2_ys1),__low2float(eta2_ya1));

        //     }
        //     else{
        //         printf("mixhalflow: %d----eta2out:%e,dfdeta2:%e,eta2lap:%e\n",i,__high2float(heta2_outd[y][x]),__high2float(hdfdeta2d[y][x]),__high2float(heta2_lapd[y][x]));
        //         printf("mixhalflow: %d----eta2:%e, xs1:%e,xa1:%e,ys1:%e,ya1:%e\n",i,__low2float(heta2d[y][x]),__high2float(eta2_xs1),__high2float(eta2_xa1),__high2float(eta2_ys1),__high2float(eta2_ya1));
        //     }
        // }
        // #endif
    }
    else{

        x_offset=x_offset+uxd2*blockIdx.x;
        int x=x_start*2+x_offset;
        int xs1=x>0?x-1:dimX-1;
        int xa1=x<dimX-1?x+1:0;
        #ifdef AMSTENCIL
        eta2_lapd[y][x]=(eta2d[y][xs1]+eta2d[y][xa1]+eta2d[ys1][x]+eta2d[ya1][x]-4.0*eta2d[y][x])/dxdy;
        #endif
        #ifdef GRAM1
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        highprecision eta2_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?eta2d[y][xs1]:__half2float(heta2d[y][xs1]);
        highprecision eta2_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?eta2d[y][xa1]:__half2float(heta2d[y][xa1]);
        highprecision eta2_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?eta2d[ys1][x]:__half2float(heta2d[ys1][x]);
        highprecision eta2_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?eta2d[ya1][x]:__half2float(heta2d[ya1][x]);
        eta2_lapd[y][x]=(eta2_xs1+eta2_xa1+eta2_ys1+eta2_ya1-4.0*eta2d[y][x])/dxdy; 
        #endif
        #ifdef GRAM2
        int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
        int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
        highprecision eta2_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?eta2d[y][xs1]:__half2float(heta2d[y][xs1]);
        highprecision eta2_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?eta2d[y][xa1]:__half2float(heta2d[y][xa1]);
        highprecision eta2_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?eta2d[ys1][x]:__half2float(heta2d[ys1][x]);
        highprecision eta2_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?eta2d[ya1][x]:__half2float(heta2d[ya1][x]);
        eta2_lapd[y][x]=(eta2_xs1+eta2_xa1+eta2_ys1+eta2_ya1-4.0*eta2d[y][x])/dxdy;
        #endif
        highprecision sum2=pow(eta1d[y][x],2)+pow(eta2d[y][x],2);
        dfdeta2d[y][x]=1.0*(-12.0*pow(eta2d[y][x],2)*(2.0-cond[y][x])+12.0*eta2d[y][x]*(1.0-cond[y][x])+12.0*eta2d[y][x]*sum2);
        eta2_outd[y][x]=eta2d[y][x]-dtime*coefl*(dfdeta2d[y][x]-0.5*coefk*eta2_lapd[y][x]);
        if(eta2_outd[y][x]>=1) eta2_outd[y][x]=1;
        else if(eta2_outd[y][x]<0)eta2_outd[y][x]=0;
        #ifdef monitor_conversion_dependent
        #ifdef AMSTENCIL
        if(x_offset==0)heta2_outd[y][x_start]=__halves2half2(__float2half(eta2_outd[y][x]),__high2half(heta2_outd[y][x_start]));
        if(x_offset==uxs1)heta2_outd[y][x_start+x_offset-uxd2]=__halves2half2(__low2half(heta2_outd[y][x_start+x_offset-uxd2]),__float2half(eta2_outd[y][x]));
        #endif
        #endif
    }
}
#endif

#ifdef monitor_conversion_independent
__global__ void monitor2_lastdata_store(highprecision *con,highprecision *con_last,int i){

    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int x_offset=threadIdx.x;
    int x_start=unitindex_x*uxd2;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    if(i%10==0){
        highprecision(*con_lastd)[dimX]=(highprecision(*)[dimX])con_last;
        con_lastd[y][x_start*2+x_offset+uxd2*blockIdx.x]=cond[y][x_start*2+x_offset+uxd2*blockIdx.x];
    }
}
__global__ void con1_conversion(int *type_curr,highprecision *dummy,lowprecision* hdummy){

    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int type;
    int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
    type=type_currd[unitindex_y][unitindex_x];
    int x_offset=threadIdx.x;
    int x_start=unitindex_x*uxd2;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    lowprecision(*hdummyd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hdummy;
    highprecision(*dummyd)[dimX]=(highprecision(*)[dimX])dummy;
    if(type==1){
        //计算点的坐标
        if(blockIdx.x==1)return;
        int x=x_start+x_offset;
        if(y_offset==0||y_offset==uys1){
            dummyd[y][x_start*2+x_offset]=__low2float(hdummyd[y][x]);
            dummyd[y][x_start*2+x_offset+uxd2]=__high2float(hdummyd[y][x]);
        }
        else if(x_offset==0)dummyd[y][x_start*2]=__low2float(hdummyd[y][x]);
        else if(x_offset==uxd2s1)dummyd[y][x_start*2+x_offset+uxd2]=__high2float(hdummyd[y][x]);
    }
    //double
    else{
        x_offset=x_offset+blockIdx.x*uxd2;
        int x=x_start*2+x_offset;
        if(x_offset==0)hdummyd[y][x_start]=__halves2half2(__float2half(dummyd[y][x]),__high2half(hdummyd[y][x_start]));
        if(x_offset==uxs1)hdummyd[y][x_start+x_offset-uxd2]=__halves2half2(__low2half(hdummyd[y][x_start+x_offset-uxd2]),__float2half(dummyd[y][x]));
    }
}
__global__ void con2_conversion(int *type_curr,highprecision *con,lowprecision *hcon,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int type;
    int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
    type=type_currd[unitindex_y][unitindex_x];
    int x_offset=threadIdx.x;
    int x_start=unitindex_x*uxd2;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    lowprecision(*hcond)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hcon;
    //half2
    if(type==1){
        if(blockIdx.x==1)return;
        int x=x_start+x_offset;
        if(y_offset==0||y_offset==uys1||(i+1)%10==0){
            cond[y][x_start*2+x_offset]=__low2float(hcond[y][x]);
            cond[y][x_start*2+x_offset+uxd2]=__high2float(hcond[y][x]);
        }
        else if(x_offset==0)cond[y][x_start*2]=__low2float(hcond[y][x]);
        else if(x_offset==uxd2s1)cond[y][x_start*2+x_offset+uxd2]=__high2float(hcond[y][x]); 
    }
    //double
    else{
        x_offset=x_offset+uxd2*blockIdx.x;
        int x=x_start*2+x_offset;
        if(x_offset==0)hcond[y][x_start]=__halves2half2(__float2half(cond[y][x]),__high2half(hcond[y][x_start]));
        if(x_offset==uxs1)hcond[y][x_start+x_offset-uxd2]=__halves2half2(__low2half(hcond[y][x_start+x_offset-uxd2]),__float2half(cond[y][x]));
    }
}
__global__ void phi1_conversion(int* type_curr,highprecision* eta1_out,lowprecision* heta1_out,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int type;
    int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
    type=type_currd[unitindex_y][unitindex_x];
    int x_offset=threadIdx.x;
    int x_start=unitindex_x*uxd2;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    highprecision(*eta1_outd)[dimX]=(highprecision(*)[dimX])eta1_out;
    lowprecision(*heta1_outd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta1_out;
    if(type==1){
        if(blockIdx.x==1)return;
        int x=x_start+x_offset;
        if(y_offset==0||y_offset==uys1||(i+1)%10==0){
            eta1_outd[y][x_start*2+x_offset]=__low2float(heta1_outd[y][x]);
            eta1_outd[y][x_start*2+x_offset+uxd2]=__high2float(heta1_outd[y][x]);
        }
        else if(x_offset==0)eta1_outd[y][x_start*2]=__low2float(heta1_outd[y][x]);
        else if(x_offset==uxd2s1)eta1_outd[y][x_start*2+x_offset+uxd2]=__high2float(heta1_outd[y][x]);
    }
    else{
        x_offset=x_offset+uxd2*blockIdx.x;
        int x=x_start*2+x_offset;
        if(x_offset==0)heta1_outd[y][x_start]=__halves2half2(__float2half(eta1_outd[y][x]),__high2half(heta1_outd[y][x_start]));
        if(x_offset==uxs1)heta1_outd[y][x_start+x_offset-uxd2]=__halves2half2(__low2half(heta1_outd[y][x_start+x_offset-uxd2]),__float2half(eta1_outd[y][x]));
    }
}
__global__ void phi2_conversion(int* type_curr,highprecision* eta2_out,lowprecision* heta2_out,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int type;
    int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
    type=type_currd[unitindex_y][unitindex_x];
    int x_offset=threadIdx.x;
    int x_start=unitindex_x*uxd2;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    lowprecision(*heta2_outd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])heta2_out;
    highprecision(*eta2_outd)[dimX]=(highprecision(*)[dimX])eta2_out;
    if(type==1){
        if(blockIdx.x==1)return;
        int x=x_start+x_offset;
        if(y_offset==0||y_offset==uys1||(i+1)%10==0){
            eta2_outd[y][x_start*2+x_offset]=__low2float(heta2_outd[y][x]);
            eta2_outd[y][x_start*2+x_offset+uxd2]=__high2float(heta2_outd[y][x]);
        }
        else if(x_offset==0)eta2_outd[y][x_start*2+x_offset]=__low2float(heta2_outd[y][x]);
        else if(x_offset==uxd2s1)eta2_outd[y][x_start*2+x_offset+uxd2]=__high2float(heta2_outd[y][x]);
    }
    else{
        x_offset=x_offset+uxd2*blockIdx.x;
        int x=x_start*2+x_offset;
        if(x_offset==0)heta2_outd[y][x_start]=__halves2half2(__float2half(eta2_outd[y][x]),__high2half(heta2_outd[y][x_start]));
        if(x_offset==uxs1)heta2_outd[y][x_start+x_offset-uxd2]=__halves2half2(__low2half(heta2_outd[y][x_start+x_offset-uxd2]),__float2half(eta2_outd[y][x]));
    }
}
#endif