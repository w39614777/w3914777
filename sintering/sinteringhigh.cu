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
typedef double highprecision;
const int dimX=512,dimY=512;
const int unitx=16,unity=16,unitdimX=dimX/unitx,unitdimY=dimY/unity,uxd2=unitx/2,uxd2s1=uxd2-1,uxs1=unitx-1,uys1=unity-1,dimXd2=dimX/2,unitNums=unitdimX*unitdimY,lowprecison_dimX=dimX/2;
const highprecision coefm=5.0,coefk=2.0,coefl=5.0,dvol=0.040,dvap=0.002,dsur=32,dgrb=1.6,dx=0.5,dy=0.5,dxdy=dx*dy,dtime=1.0e-4;
highprecision R1=50,R2=R1/2;
highprecision Ry1=240,Ry2=Ry1+R1+R2,Rx1=dimX/2+8;
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
__global__ void con1(highprecision* hcon,highprecision* hcon_lap,highprecision* heta1,highprecision* heta2,highprecision* hdfdcon,highprecision* hdummy){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    highprecision(*hcond)[dimX]=(highprecision(*)[dimX])hcon;
    highprecision(*hcon_lapd)[dimX]=(highprecision(*)[dimX])hcon_lap;
    highprecision(*heta1d)[dimX]=(highprecision(*)[dimX])heta1;
    highprecision(*heta2d)[dimX]=(highprecision(*)[dimX])heta2;
    highprecision(*hdfdcond)[dimX]=(highprecision(*)[dimX])hdfdcon;
    highprecision(*hdummyd)[dimX]=(highprecision(*)[dimX])hdummy;
    hcon_lapd[y][x]=(hcond[y][xs1]+hcond[y][xa1]+hcond[ys1][x]+hcond[ya1][x]-4.0*hcond[y][x])/dxdy;
    highprecision sum2=pow(heta1d[y][x],2)+pow(heta2d[y][x],2);
    highprecision sum3=pow(heta1d[y][x],3)+pow(heta2d[y][x],3);
    hdfdcond[y][x]=1.0*(2.0*hcond[y][x]+4.0*sum3-6.0*sum2)-2.0*16.0*pow(hcond[y][x],2)*(1.0-hcond[y][x])+2.0*16.0*hcond[y][x]*pow(1.0-hcond[y][x],2);
    hdummyd[y][x]=hdfdcond[y][x]-0.5*coefm*hcon_lapd[y][x];
}
__global__ void con2(highprecision* dummy,highprecision* dummy_lap,highprecision* con,highprecision* eta1,highprecision* eta2){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
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
}
__global__ void phi1(highprecision* eta1,highprecision* eta1_out,highprecision* eta2,highprecision* eta1_lap,highprecision* dfdeta1,highprecision* con){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
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

__global__ void phi2(highprecision* eta2,highprecision* eta2_out,highprecision* eta1,highprecision* eta2_lap,highprecision* dfdeta2,highprecision* con){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    highprecision(*eta2_outd)[dimX]=(highprecision(*)[dimX])eta2_out;
    highprecision(*eta2_lapd)[dimX]=(highprecision(*)[dimX])eta2_lap;
    highprecision(*dfdeta2d)[dimX]=(highprecision(*)[dimX])dfdeta2;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;

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

int main(void){
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
    dim3 blocks(32,32);
    dim3 grids(dimX/32,dimY/32);
    con1<<<grids,blocks>>>(con,con_lap,eta1,eta2,dfdcon,dummy);
    cudaDeviceSynchronize();
    con2<<<grids,blocks>>>(dummy,dummy_lap,con,eta1,eta2);
    cudaDeviceSynchronize();
    phi1<<<grids,blocks>>>(eta1,eta1_out,eta2,eta1_lap,dfdeta1,con);
    cudaDeviceSynchronize();
    phi2<<<grids,blocks>>>(eta2,eta2_out,eta1,eta2_lap,dfdeta2,con);
    cudaDeviceSynchronize();
    swap(eta1,eta1_out);
    swap(eta2,eta2_out);
    CHECK_ERROR(cudaFree(con));CHECK_ERROR(cudaFree(con_lap));
    CHECK_ERROR(cudaFree(eta1));CHECK_ERROR(cudaFree(eta1_lap));
    CHECK_ERROR(cudaFree(eta2));CHECK_ERROR(cudaFree(eta2_lap));
    CHECK_ERROR(cudaFree(eta1_out));CHECK_ERROR(cudaFree(eta2_out));
    CHECK_ERROR(cudaFree(dummy));CHECK_ERROR(cudaFree(dummy_lap));
    CHECK_ERROR(cudaFree(dfdcon));CHECK_ERROR(cudaFree(dfdeta1));CHECK_ERROR(cudaFree(dfdeta2));
    return 0;
}