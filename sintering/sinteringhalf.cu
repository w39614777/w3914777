#include "stdio.h"
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <mma.h>
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <math.h>
#include<thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
using namespace std;
using namespace nvcuda;
const int dimX=512,dimY=512;
const float coefm=5.0,coefk=2.0,coefl=5.0,dvol=0.040,dvap=0.002,dsur=16.0,dgrb=1.6,dx=0.2,dy=0.2,dxdy=dx*dy,dtime=1.0e-4;
int it=1;
float  R1=50.0,R2=25.0;//两种粒子的半径平方大小
float  Rx1=dimX/2,Rx2=Rx1,Ry1=200.0,Ry2=275.0;//两种粒子的中心坐标
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
__global__ void half_con1(half* hcon,half* hcon_lap,half* heta1,half* heta2,half* hdfdcon,half* hdummy,
half hfour,half hdxdy,half hsix,half hzpfive,half hone,half htwo,half hsixteen,half hcoefm){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    half(*hcond)[dimX]=(half(*)[dimX])hcon;
    half(*hcon_lapd)[dimX]=(half(*)[dimX])hcon_lap;
    half(*heta1d)[dimX]=(half(*)[dimX])heta1;
    half(*heta2d)[dimX]=(half(*)[dimX])heta2;
    half(*hdfdcond)[dimX]=(half(*)[dimX])hdfdcon;
    half(*hdummyd)[dimX]=(half(*)[dimX])hdummy;
    hcon_lapd[y][x]=(hcond[y][xs1]+hcond[y][xa1]+hcond[ys1][x]+hcond[ya1][x]-hfour*hcond[y][x])/hdxdy;
    half sum2=(heta1d[y][x]*heta1d[y][x])+(heta2d[y][x]*heta2d[y][x]);
    half sum3=(heta1d[y][x]*heta1d[y][x]*heta1d[y][x])+(heta2d[y][x]*heta2d[y][x]*heta2d[y][x]);
    hdfdcond[y][x]=hone*(htwo*hcond[y][x]+hfour*sum3-hsix*sum2)-htwo*hsixteen*(hcond[y][x]*hcond[y][x])*(hone-hcond[y][x])+htwo*hsixteen*hcond[y][x]*((hone-hcond[y][x])*(hone-hcond[y][x]));
    hdummyd[y][x]=hdfdcond[y][x]-hzpfive*hcoefm*hcon_lapd[y][x];
}
__global__ void half_con2(half* hdummy,half* hdummy_lap,half* hcon,half* heta1,half* heta2,
half hfour,half hdxdy,half hone,half htwo,half hsix,half hten,half hfifteen,half hdgrb,half hdsur,half hdtime,half hdvap,half hdvol,half highbound,half lowbound){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    half(*hdummyd)[dimX]=(half(*)[dimX])hdummy;
    half(*hdummy_lapd)[dimX]=(half(*)[dimX])hdummy_lap;
    half(*hcond)[dimX]=(half(*)[dimX])hcon;
    half(*heta1d)[dimX]=(half(*)[dimX])heta1;
    half(*heta2d)[dimX]=(half(*)[dimX])heta2;
    hdummy_lapd[y][x]=(hdummyd[y][xs1]+hdummyd[y][xa1]+hdummyd[ys1][x]+hdummyd[ya1][x]-hfour*hdummyd[y][x])/hdxdy;
    half hphi=(hcond[y][x]*hcond[y][x]*hcond[y][x])*(hten-hfifteen*hcond[y][x]+hsix*(hcond[y][x]*hcond[y][x])); //插值函数
    half hsum=heta1d[y][x]*heta2d[y][x]*htwo;
    half hmobil=hdvol*hphi+hdvap*(hone-hphi)+hdsur*hcond[y][x]*(hone-hcond[y][x])+hdgrb*hsum;
    hcond[y][x]=hcond[y][x]+hdtime*hmobil*hdummy_lapd[y][x];
    hcond[y][x]=__hmin(highbound,hcond[y][x]);
    hcond[y][x]=__hmax(lowbound,hcond[y][x]);
}
__global__ void half_phi1(half* heta1,half* heta1_out,half* heta2,half* heta1_lap,half* hdfdeta1,half* hcon,
half hfour,half hdxdy,half hzpfive,half hone,half htwo,half htwelve,half hdtime,half hcoefl,half hcoefk,half highbound,half lowbound){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    half(*heta1d)[dimX]=(half(*)[dimX])heta1;
    half(*heta2d)[dimX]=(half(*)[dimX])heta2;
    half(*heta1_outd)[dimX]=(half(*)[dimX])heta1_out;
    half(*heta1_lapd)[dimX]=(half(*)[dimX])heta1_lap;
    half(*hdfdeta1d)[dimX]=(half(*)[dimX])hdfdeta1;
    half(*hcond)[dimX]=(half(*)[dimX])hcon;
    heta1_lapd[y][x]=(heta1d[y][xs1]+heta1d[y][xa1]+heta1d[ys1][x]+heta1d[ya1][x]-hfour*heta1d[y][x])/hdxdy;
    //自由能对相求导
    half hsum2=(heta1d[y][x]*heta1d[y][x])+(heta2d[y][x]*heta2d[y][x]);
    hdfdeta1d[y][x]=hone*(-htwelve*(heta1d[y][x]*heta1d[y][x])*(htwelve-hcond[y][x])+htwelve*heta1d[y][x]*(hone-hcond[y][x])+htwelve*heta1d[y][x]*hsum2);
    heta1_outd[y][x]=heta1d[y][x]-hdtime*hcoefl*(hdfdeta1d[y][x]-hzpfive*hcoefk*heta1_lapd[y][x]);
    heta1_outd[y][x]=__hmin(highbound,heta1_outd[y][x]);
    heta1_outd[y][x]=__hmax(lowbound,heta1_outd[y][x]);
}

__global__ void half_phi2(half* heta2,half* heta2_out,half* heta1,half* heta2_lap,half* hdfdeta2,half* hcon,
half hfour,half hdxdy,half hzpfive,half hone,half htwo,half htwelve,half hdtime,half hcoefl,half hcoefk,half highbound,half lowbound){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    half(*heta1d)[dimX]=(half(*)[dimX])heta1;
    half(*heta2d)[dimX]=(half(*)[dimX])heta2;
    half(*heta2_outd)[dimX]=(half(*)[dimX])heta2_out;
    half(*heta2_lapd)[dimX]=(half(*)[dimX])heta2_lap;
    half(*hdfdeta2d)[dimX]=(half(*)[dimX])hdfdeta2;
    half(*hcond)[dimX]=(half(*)[dimX])hcon;

    heta2_lapd[y][x]=(heta2d[y][xs1]+heta2d[y][xa1]+heta2d[ys1][x]+heta2d[ya1][x]-hfour*heta2d[y][x])/hdxdy;
    //自由能对相求导
    half hsum2=(heta1d[y][x]*heta1d[y][x])+(heta2d[y][x]*heta2d[y][x]);
    hdfdeta2d[y][x]=hone*(-htwelve*(heta2d[y][x]*heta2d[y][x])*(htwo-hcond[y][x])+htwelve*heta2d[y][x]*(hone-hcond[y][x])+htwelve*heta2d[y][x]*hsum2);
    heta2_outd[y][x]=heta2d[y][x]-hdtime*hcoefl*(hdfdeta2d[y][x]-hzpfive*hcoefk*heta2_lapd[y][x]);
    heta2_outd[y][x]=__hmin(highbound,heta2_outd[y][x]);
    heta2_outd[y][x]=__hmax(lowbound,heta2_outd[y][x]);
}

int main(void){
    half *hcon,*heta1,*heta2,*heta1_lap,*heta2_lap,*hcon_lap,*hdummy,*hdummy_lap,*hdfdcon,*hdfdeta1,*hdfdeta2,*heta1_out,*heta2_out;
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
    half hcoefm=__float2half(coefm);
    half hcoefk=__float2half(coefk);
    half hcoefl=__float2half(coefl);
    half hdvol=__float2half(dvol);
    half hdvap=__float2half(dvap);
    half hdsur=__float2half(dsur);
    half hdgrb=__float2half(dgrb);
    half hdx=__float2half(dx);
    half hdy=__float2half(dy);
    half hdxdy=__float2half(dxdy);
    half hdtime=__float2half(dtime);
    half hzpfive=__float2half(0.5);
    half hone=__float2half(1.0);
    half htwo=__float2half(2.0);
    half hfour=__float2half(4.0);
    half hsix=__float2half(6.0);
    half hten=__float2half(10.0);
    half htwelve=__float2half(12.0);
    half hfifteen=__float2half(15.0);
    half hsixteen=__float2half(16.0);
    half highbound=__float2half(1);
    half lowbound=__float2half(0.0);
    for(int y=1;y<=dimY;y++){
        for(int x=1;x<=dimX;x++){
            float dis1=sqrt(pow(x-Rx1,2)+pow(y-Ry1,2));
            float dis2=sqrt(pow(x-Rx1,2)+pow(y-Ry2,2));
            if(dis1<=R1){
                hcon[(y-1)*dimX+x-1]=(half)1;
                heta1[(y-1)*dimX+x-1]=(half)1;
            }
            if(dis2<=R2){
                hcon[(y-1)*dimX+x-1]=(half)1;
                heta2[(y-1)*dimX+x-1]=(half)1;
                heta1[(y-1)*dimX+x-1]=(half)0.0;
            }
        }
    }
    dim3 blocks(32,32);
    dim3 grids(dimX/32,dimY/32);
    half_con1<<<grids,blocks>>>(hcon,hcon_lap,heta1,heta2,hdfdcon,hdummy,hfour,hdxdy,hsix,hzpfive,hone,htwo,hsixteen,hcoefm);
    cudaDeviceSynchronize();
    half_con2<<<grids,blocks>>>(hdummy,hdummy_lap,hcon,heta1,heta2,hfour,hdxdy,hone,htwo,hsix,hten,hfifteen,hdgrb,hdsur,hdtime,hdvap,hdvol,highbound,lowbound);
    cudaDeviceSynchronize();
    half_phi1<<<grids,blocks>>>(heta1,heta1_out,heta2,heta1_lap,hdfdeta1,hcon,hfour,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound);
    cudaDeviceSynchronize();
    half_phi2<<<grids,blocks>>>(heta2,heta2_out,heta1,heta2_lap,hdfdeta2,hcon,hfour,hdxdy,hzpfive,hone,htwo,htwelve,hdtime,hcoefl,hcoefk,highbound,lowbound);
    cudaDeviceSynchronize();
    swap(heta1,heta1_out);
    swap(heta2,heta2_out);
    CHECK_ERROR(cudaFree(hcon));CHECK_ERROR(cudaFree(hcon_lap));
    CHECK_ERROR(cudaFree(heta1));CHECK_ERROR(cudaFree(heta1_lap));
    CHECK_ERROR(cudaFree(heta2));CHECK_ERROR(cudaFree(heta2_lap));
    CHECK_ERROR(cudaFree(heta1_out));CHECK_ERROR(cudaFree(heta2_out));
    CHECK_ERROR(cudaFree(hdummy));CHECK_ERROR(cudaFree(hdummy_lap));
    CHECK_ERROR(cudaFree(hdfdcon));CHECK_ERROR(cudaFree(hdfdeta1));CHECK_ERROR(cudaFree(hdfdeta2));
    return 0;
}