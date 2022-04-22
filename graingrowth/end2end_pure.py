import os
import sys
for datatype in ["double","float"]:
    for para in range(1,6):
        os.system("rm -rf ./time/"+datatype+"/para"+str(para)+"/PURE")
for datatype in ["double","float"]:
    file=open("./paras/total.h","r+");
    flist=file.readlines()
    file.close()
    flist[17]="   typedef "+datatype+" highprecision;\n"
    flist[18]= "   int timesteps=500;\n"
    flist[26]="   #define End2end\n"
    flist[24]="   #define PURE\n"
    file=open("./paras/total.h","w+")
    file.writelines(flist)
    file.close()
    for para in range(1,6):
        resultpath="./result/end2end/"+datatype+"/para"+str(para)+"/PURE/"
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)
        if not os.path.exists(resultpath+"500.csv"):
            os.mknod(resultpath+"500.csv")
        timepath="./time/"+datatype+"/para"+str(para)+"/PURE/"
        if not os.path.exists(timepath):
            os.makedirs(timepath)
        if not os.path.exists(timepath+"time.csv"):
            os.mknod(timepath+"time.csv")
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
        if os.system("nvcc purehigh.cu -o purehigh --std=c++11 -arch=sm_80 -w")==0:
            listtime=[]
            for i in range(5):
                os.system("./purehigh "+str(i)+" "+resultpath+"500.csv")
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
            sys.exit(1)