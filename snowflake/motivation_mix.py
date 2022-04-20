import os
import sys
from time import time


ratios=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for datatype in ["double"]:
    for timestep in ['500','1000','1500','2000']:
        for strategy in ["AMSTENCIL",'GRAM1','GRAM2']:
            for ratio in   ratios:
                file=open("./paras/total.h","r+");
                flist=file.readlines()
                file.close()
                flist[17]="   typedef "+datatype+" highprecision;\n"
                flist[18]="   int timesteps="+timestep+";\n"
                flist[22]="  const highprecision ratio="+str(ratio)+";\n"
                flist[24]="   #define "+strategy+"\n"
                flist[26]="   #define Motivation\n"
                flist[45]="   #define monitor_conversion_dependent\n"
                file=open("./paras/total.h","w+")
                file.writelines(flist)
                file.close()
                for para in range(1,6):
                    file=open("./function.h","r+")
                    flist=file.readlines()
                    file.close()
                    flist[2]="    #include\"./paras/para"+str(para)+".h\"\n"
                    file=open("./function.h","w+")
                    file.writelines(flist)
                    file.close()
                    file=open("./tools.h","r+")
                    flist=file.readlines()
                    file.close()
                    flist[2]="    #include\"./paras/para"+str(para)+".h\"\n"
                    file=open("./tools.h","w+")
                    file.writelines(flist)
                    file.close()              
                    if strategy=="AMSTENCIL":
                        file=open("./paras/total.h","r+");
                        flist=file.readlines()
                        file.close()
                        for monitor in range(1,3):
                            flist[27]="   #define Monitor"+str(monitor)+"\n"
                            file=open("./paras/total.h","w+")
                            file.writelines(flist)
                            file.close()
                            resultpath="./result/motivation/"+datatype+"/para"+str(para)+"/"+strategy+"/monitor"+str(monitor)+"/"+timestep+"/"
                            if not os.path.exists(resultpath):
                                os.makedirs(resultpath)
                            if not os.path.exists(resultpath+str(ratio)+".csv"):
                                os.mknod(resultpath+str(ratio)+".csv")
                            if os.system("nvcc motivation_mix.cu -o motivation_mix --std=c++11 -arch=sm_80 -w")==0:
                                os.system("./motivation_mix "+resultpath+str(ratio)+".csv")
                            else:
                                print(strategy,datatype,str(ratio),str(para))
                                sys.exit(1)
                    else:
                        resultpath="./result/motivation/"+datatype+"/para"+str(para)+"/"+strategy+"/"+timestep+"/"
                        if not os.path.exists(resultpath):
                            os.makedirs(resultpath)
                        if not os.path.exists(resultpath+str(ratio)+".csv"):
                            os.mknod(resultpath+str(ratio)+".csv")
                        if os.system("nvcc motivation_mix.cu -o motivation_mix --std=c++11 -arch=sm_80 -w")==0:
                            os.system("./motivation_mix "+resultpath+str(ratio)+".csv")
                        else:
                            print(strategy,datatype,str(ratio),str(para))
                            sys.exit(1)                                        