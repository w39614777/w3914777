import os
import sys
for datatype in ["double","float"]:
    for timestep in ['500','1000','1500','2000']:
        file=open("./paras/total.h","r+");
        flist=file.readlines()
        file.close()
        flist[17]="   typedef "+datatype+" highprecision;\n"
        flist[18]="   int timesteps="+timestep+";\n"
        flist[24]="   #define PURE\n"
        flist[26]="   #define Motivation\n"
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
            resultpath="./result/motivation/"+datatype+"/para"+str(para)+"/PURE/"+timestep+"/"
            if not os.path.exists(resultpath):
                os.makedirs(resultpath)
            if not os.path.exists(resultpath+"500.csv"):
                os.mknod(resultpath+"500.csv")
            if os.system("nvcc purehigh.cu -o purehigh --std=c++11 -arch=sm_80 -w")==0:
                os.system("./purehigh "+resultpath+timestep+".csv")
            else:
                print(datatype,str(para))
                sys.exit(1)                                        