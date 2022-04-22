import os
import sys
for datatype in ["double","float"]:
    for strategy in ['AMSTENCIL','GRAM1',"GRAM2","PURE"]:
        os.system("rm -rf ./time/500/"+datatype+"/"+strategy)
        print("rm -rf ./time/500/"+datatype+"/"+strategy)
strategys=['AMSTENCIL','GRAM1',"GRAM2","PURE"]
timestep="500"
get_time=1
get_result=0
thresholds=None
ratios=[]
for i in range(1,101):
    ratios.append(str(round(0.01*i,2)))
datatypes=["float","double"]
for datatype in datatypes:
    for strategy in strategys:
        for para_index in range(1,6):
            time_path=""
            if get_time:
                time_path="./time/"+timestep+"/"+datatype+"/"+strategy+"/"

            fcu=open("function.h","r+")
            flist=fcu.readlines()
            fcu.close()
            flist[0]="#include \"./paras/para"+str(para_index)+".h\"\n"
            flist[18]="#define "+strategy+"\n"
            flist[21]="#define End2end\n"
            if get_time:
                flist[20]="#define GET_TIME\n"
            else:
                flist[20]="#define GET_RESULT\n"
            fcu=open("function.h","w+")
            fcu.writelines(flist)
            fcu.close()
            fpara=open("./paras/para"+str(para_index)+".h","r+")
            fparalist=fpara.readlines()
            fpara.close()
            fparalist[0]="typedef "+datatype+" highprecision;\n"
            fparalist[1]="int timesteps="+timestep+";\n"
            if strategy=="AMSTENCIL":
                for monitor in [1,2]:
                    if monitor==1:
                        thresholds=[0.000001,0.00001,0.0001,0.001,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
                    if monitor==2:
                        thresholds=[1e-20,1e-19,1e-18,1e-17,1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
                    fcu=open("function.h","r+")
                    flist=fcu.readlines()
                    fcu.close()
                    flist[22]='#define Monitor'+str(monitor)+'\n'
                    fcu=open("function.h","w+")
                    fcu.writelines(flist)
                    fcu.close()
                    time_path="./time/"+timestep+"/"+datatype+"/"+strategy+"/"
                    time_path=time_path+"monitor"+str(monitor)+"/para"+str(para_index)+"/"
                    for threshold in thresholds:
                        fparalist[7]="const highprecision threshold="+str(threshold)+";\n"
                        fpara=open("./paras/para"+str(para_index)+".h","w+")
                        fpara.writelines(fparalist)
                        fpara.close()
                        if os.system("nvcc main.cu -o main --std=c++11 -arch=sm_80 -w")==0:
                            if  get_result:
                                os.system("./main "+str(para_index)+" "+datatype)
                            else:
                                listtime=[str(threshold)]
                                for i in range(5):
                                    os.system("./main")
                                    ftime=open("time_tmp.csv","r+")
                                    time=ftime.read()
                                    ftime.close()
                                    listtime.append(time)
                                if not os.path.exists(time_path):
                                    os.makedirs(time_path)
                                if os.path.exists(time_path+"time.csv")==False:
                                    os.mknod(time_path+"time.csv") 
                                f=open(time_path+"time.csv","r+")
                                this_time_list=f.readlines()
                                f.close()
                                this_time_list.append(",".join(listtime)+"\n")
                                f=open(time_path+"time.csv","w+")
                                f.writelines(this_time_list)
                                f.close()
                        else:
                            print(strategy,str(threshold),str(para_index))
                            sys.exit(1)
            elif strategy=="GRAM1" or strategy=="GRAM2":
                time_path=time_path+"para"+str(para_index)+"/"
                for ratio in ratios:
                    print(datatype,str(para_index),ratio)
                    fparalist[8]="const highprecision ratio="+ratio+";\n"
                    fpara=open("./paras/para"+str(para_index)+".h","w+")
                    fpara.writelines(fparalist)
                    fpara.close()
                    if os.system("nvcc main.cu -o main --std=c++11 -arch=sm_80 -w")==0:
                        if not get_time:
                            os.system("./main "+str(para_index)+" "+datatype)
                        else:
                            listtime=[str(ratio)]
                            for i in range(5):
                                os.system("./main")
                                ftime=open("time_tmp.csv","r+")
                                time=ftime.read()
                                ftime.close()
                                listtime.append(time)
                            if not os.path.exists(time_path):
                                os.makedirs(time_path)
                            if os.path.exists(time_path+"time.csv")==False:
                                os.mknod(time_path+"time.csv") 
                            f=open(time_path+"time.csv","r+")
                            this_time_list=f.readlines()
                            f.close()
                            this_time_list.append(",".join(listtime)+"\n")
                            f=open(time_path+"time.csv","w+")
                            f.writelines(this_time_list)
                            f.close()                            
                    else:
                        print(strategy,ratio,str(para_index))
                        sys.exit(1)                    

            else:
                time_path=time_path+"para"+str(para_index)+"/"
                fpara=open("./paras/para"+str(para_index)+".h","w+")
                fpara.writelines(fparalist)
                fpara.close()
                if os.system("nvcc main.cu -o main --std=c++11 -arch=sm_80 -w")==0:
                    if not get_time:
                        os.system("./main "+str(para_index)+" "+datatype)
                    else:
                        listtime=[]
                        for i in range(5):
                            os.system("./main")
                            ftime=open("time_tmp.csv","r+")
                            time=ftime.read()
                            ftime.close()
                            listtime.append(time)
                        if not os.path.exists(time_path):
                            os.makedirs(time_path)
                        if os.path.exists(time_path+"time.csv")==False:
                            os.mknod(time_path+"time.csv") 
                        f=open(time_path+"time.csv","r+")
                        this_time_list=f.readlines()
                        f.close()
                        this_time_list.append(",".join(listtime)+"\n")
                        f=open(time_path+"time.csv","w+")
                        f.writelines(this_time_list)
                        f.close()     
                else:
                    print(strategy,str(para_index),os.system("nvcc main.cu -o main --std=c++11 -arch=sm_80"))
                    sys.exit(1)   