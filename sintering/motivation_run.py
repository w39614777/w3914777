import os
import sys
strategys=['AMSTENCIL','GRAM1',"GRAM2","PURE"]
ratios=[]
for i in range(1,10):
    ratios.append('%.1f'%(0.1*i))
datatypes=["double"]
timesteps=['500','1000','1500','2000']
for datatype in datatypes:
    for strategy in strategys:
        for para_index in range(1,6):
            for timestep in timesteps:
                fh=open("function.h","r+")
                functionlist=fh.readlines()
                fh.close()
                functionlist[0]="#include \"./paras/para"+str(para_index)+".h\"\n"
                functionlist[18]="#define "+strategy+"\n"
                functionlist[20]="#define GET_RESULT\n"
                functionlist[21]="#define Motivation\n"                
                fh=open("function.h","w+")
                fh.writelines(functionlist)
                fh.close()
                fpara=open("./paras/para"+str(para_index)+".h","r+")
                paralist=fpara.readlines()
                fpara.close()
                paralist[0]="typedef "+datatype+" highprecision;\n"
                paralist[1]="int timesteps="+timestep+";\n"
                if strategy=="PURE":
                    fpara=open("./paras/para"+str(para_index)+".h","w+")
                    fpara.writelines(paralist)
                    fpara.close()
                    if os.system("nvcc main.cu -o main --std=c++11 -arch=sm_80 -w")==0:
                        os.system("./main "+str(para_index)+" "+datatype)
                    else:
                        sys.exit(1)
                else:
                    for ratio in ratios:
                        paralist[8]="const highprecision ratio="+ratio+";\n"
                        fpara=open("./paras/para"+str(para_index)+".h","w+")
                        fpara.writelines(paralist)
                        fpara.close()
                        if(strategy=='AMSTENCIL'):
                            for monitor in range(1,3):
                                fh=open("function.h","r+")
                                functionlist=fh.readlines()
                                fh.close()
                                functionlist[22]="#define Monitor"+str(monitor)+"\n"
                                fh=open("function.h","w+")
                                fh.writelines(functionlist)
                                fh.close()
                                if os.system("nvcc main.cu -o main --std=c++11 -arch=sm_80 -w")==0:
                                    os.system("./main "+str(para_index)+" "+datatype)
                                else:
                                    sys.exit(1)                                  
                        else:
                            if os.system("nvcc main.cu -o main --std=c++11 -arch=sm_80 -w")==0:
                                os.system("./main "+str(para_index)+" "+datatype)
                            else:
                                sys.exit(1)                          