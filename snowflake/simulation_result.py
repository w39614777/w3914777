import os
import sys
if not os.path.exists("./simulation_result/"):
    os.makedirs("./simulation_result/")
file=open("function.h","r+")
flist=file.readlines()
file.close()
flist[2]='    #include"./paras/para2.h"\n'
file=open("function.h","w+")
file.writelines(flist)
file.close()
file=open("tools.h","r+")
flist=file.readlines()
file.close()
flist[2]='    #include"./paras/para2.h"\n'
file=open("tools.h","w+")
file.writelines(flist)
file.close()
# FP64
file=open("./paras/total.h","r+")
flist=file.readlines()
file.close()
flist[17]="   typedef double highprecision;\n"
flist[18]="   int timesteps=1000;\n"
flist[24]="   #define PURE\n"
flist[26]="   #define End2end\n"
file=open("./paras/total.h","w+")
file.writelines(flist)
file.close()
if os.system("nvcc purehigh.cu -o purehigh --std=c++11 -arch=sm_80")==0:
    os.system("./purehigh 4 ./simulation_result/FP64.csv")
else:
    print("fp64")

# FP16
file=open("./paras/total.h","r+")
flist=file.readlines()
file.close()
flist[18]="   int timesteps=1000;\n"
flist[36]="   #define HALF2\n"
file=open("./paras/total.h","w+")
file.writelines(flist)
file.close()
if os.system("nvcc purelow.cu -o purelow --std=c++11 -arch=sm_80")==0:
    os.system("./purelow 4 ./simulation_result/FP16.csv")
else:
    print("fp16")


# AMPStencil-spa
file=open("./paras/total.h","r+")
flist=file.readlines()
file.close()
flist[17]="   typedef double highprecision;\n"
flist[18]="   int timesteps=1000;\n"
flist[21]="   const highprecision threshold=0.1;\n"
flist[24]="   #define   AMSTENCIL\n"
flist[26]="   #define End2end\n"
flist[27]="   #define Monitor1\n"
flist[45]="   #define monitor_conversion_dependent\n"
file=open("./paras/total.h","w+")
file.writelines(flist)
file.close()
if os.system("nvcc end2end_mix.cu -o end2end_mix --std=c++11 -arch=sm_80")==0:
    os.system("./end2end_mix 4 ./simulation_result/AMPStencil-spa.csv")
else:
    print("AMPStencil-spa")
# AMPStencil-tem
file=open("./paras/total.h","r+")
flist=file.readlines()
file.close()
flist[17]="   typedef double highprecision;\n"
flist[18]="   int timesteps=1000;\n"
flist[21]="   const highprecision threshold=1e-4;\n"
flist[24]="   #define   AMSTENCIL\n"
flist[26]="   #define End2end\n"
flist[27]="   #define Monitor2\n"
flist[45]="   #define monitor_conversion_dependent\n"
file=open("./paras/total.h","w+")
file.writelines(flist)
file.close()
if os.system("nvcc end2end_mix.cu -o end2end_mix --std=c++11 -arch=sm_80")==0:
    os.system("./end2end_mix 4 ./simulation_result/AMPStencil-tem.csv")
else:
    print("AMPStencil-tem")