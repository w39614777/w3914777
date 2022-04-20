import os
import sys
for datatype in ["double","float"]:
    for para in range(1,6):
        os.system("rm -rf ./time/"+datatype+"/para"+str(para)+"/without_M1_monitor")
        os.system("rm -rf ./time/"+datatype+"/para"+str(para)+"/without_M2_monitor")
for datatype in ["double","float"]:
    thresholds=None
    for monitor in range(1,3):
        if monitor==1:
            thresholds=[0.000001,0.00001,0.0001,0.001,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
        if monitor==2:
            thresholds=[1e-20,1e-19,1e-18,1e-17,1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99] 
        for threshold in thresholds:
            file=open("./paras/total.h","r+");
            flist=file.readlines()
            file.close()
            flist[17]="   typedef "+datatype+" highprecision;\n"
            flist[18]= "   int timesteps=500;\n"
            flist[21]="   const highprecision threshold="+str(threshold)+";\n"
            flist[24]="   #define AMSTENCIL\n"
            flist[26]="   #define End2end\n"
            flist[27]="   #define Monitor"+str(monitor)+"\n"
            flist[45]="   #define monitor_conversion_independent\n"
            file=open("./paras/total.h","w+")
            file.writelines(flist)
            file.close()
            for para in range(1,6): 
                timepath="./time/"+datatype+"/para"+str(para)+"/without_M"+str(monitor)+"_monitor/"
                if not os.path.exists(timepath):
                    os.makedirs(timepath)
                if not os.path.exists(timepath+"time.csv"):
                    os.mknod(timepath+"time.csv")
                print(timepath+"time.csv",str(threshold))
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
                if os.system("nvcc monitor_independent_withoutmonitor.cu -o monitor_independent_withoutmonitor --std=c++11 -arch=sm_80")==0:
                    listtime=[str(threshold)]
                    for i in range(5):
                        os.system("./monitor_independent_withoutmonitor")
                        ftime=open("time_tmp.csv","r+")
                        time=ftime.read()
                        ftime.close()
                        listtime.append(time)
                    f=open(timepath+"time.csv","r+")
                    this_time_list=f.readlines()
                    f.close()
                    this_time_list.append(",".join(listtime)+"\n")
                    f=open(timepath+"time.csv","w+")
                    f.writelines(this_time_list)
                    f.close()
                else:
                    print(datatype,str(monitor),str(threshold),str(para))
                    sys.exit(1)                        