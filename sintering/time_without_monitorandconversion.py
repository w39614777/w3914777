import os
import sys
for datatype in ["double","float"]:
    for strategy in ['AMSTENCIL','GRAM1',"GRAM2","PURE"]:
        os.system("rm -rf ./time/500/"+datatype+"/without_M1_conversion")
thresholds=None
monitors=[1,2]
datatypes=["float","double"]
timestep="500"
for monitor in monitors:
    if monitor==1:
        thresholds=[0.000001,0.00001,0.0001,0.001,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
    if monitor==2:
        thresholds=[1e-20,1e-19,1e-18,1e-17,1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    for datatype in datatypes:
        for para_index in range(1,6):
            fcu=open("function.h","r+")
            flist=fcu.readlines()
            fcu.close()
            flist[0]="#include \"./paras/para"+str(para_index)+".h\"\n"
            flist[18]="#define AMSTENCIL\n"
            flist[21]="// #define End2end\n"
            flist[20]="#define monitor_conversion_independent\n"
            if monitor==1:
                flist[22]="#define Monitor1\n"
            if monitor==2:
                flist[22]="#define Monitor2\n"
            fcu=open("function.h","w+")
            fcu.writelines(flist)
            fcu.close()
            time_path='time/'+timestep+"/"+datatype+"/without_M1_conversion/para"+str(para_index)+"/"
            if not os.path.exists(time_path):
                os.makedirs(time_path)
            if not os.path.exists(time_path+"time.csv"):
                os.mknod(time_path+"time.csv")
            for threshold in thresholds:
                fpara=open("./paras/para"+str(para_index)+".h","r+")
                fparalist=fpara.readlines()
                fpara.close()
                fparalist[0]="typedef "+datatype+" highprecision;\n"
                fparalist[1]="int timesteps="+timestep+";\n"
                fparalist[7]="const highprecision threshold="+str(threshold)+";\n"
                fpara=open("./paras/para"+str(para_index)+".h","w+")
                fpara.writelines(fparalist)
                fpara.close()
                if os.system("nvcc main.cu -o main --std=c++11 -arch=sm_80 -w")==0:
                    listtime=[str(threshold)]
                    for i in range(5):
                        os.system("./main")
                        ftime=open("time_tmp.csv","r+")
                        time=ftime.read()
                        ftime.close()
                        listtime.append(time)
                    f=open(time_path+"time.csv","r+")
                    this_time_list=f.readlines()
                    f.close()
                    this_time_list.append(",".join(listtime)+"\n")
                    f=open(time_path+"time.csv","w+")
                    f.writelines(this_time_list)
                    f.close()
                    print(datatype,para_index,threshold)
                else:
                    print("error--",datatype,para_index,threshold)
                    sys.exit(1)
            