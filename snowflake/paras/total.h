#ifndef TOTAL_H
   #define TOTAL_H
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
   typedef float highprecision;
   int timesteps=500;
   const int dimX=512,dimY=512;
   const int unitx=16,unity=16,unitdimX=dimX/unitx,unitdimY=dimY/unity,uxd2=unitx/2,uxd2s1=uxd2-1,uxs1=unitx-1,uys1=unity-1,dimXd2=dimX/2,unitNums=unitdimX*unitdimY;
   const highprecision threshold=0.99;
  const highprecision ratio=1.0;
   // #define PRINT_INFO
   #define AMSTENCIL
   // define GET_TIME表示要获取程序的执行时间，GET_RESULT表示获取程序的结果,GET_MonitorTIME表示获取amstencil的monitor时间,monitor_conversion_independent表示获取不加monitor和type conversion的时间
   #define End2end
   #define Monitor2
   #ifdef AMSTENCIL
      typedef half2 lowprecision;
      const int lowprecison_dimX=dimXd2;
   #else
      typedef half lowprecision;
      const int lowprecison_dimX=dimX;
   #endif
   // 在pure half或者 pure half2中
   #define HALF2
   #ifdef HALF
      typedef half purelowprecision;
      const int purelowprecision_dimX=dimX;
   #endif
   #ifdef HALF2
      typedef half2 purelowprecision;
      const int purelowprecision_dimX=dimXd2;
   #endif
   #define monitor_conversion_independent
   #define pi 3.1415926
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
#endif
