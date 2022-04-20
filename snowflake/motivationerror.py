import pandas as pd
import numpy as np
import os
timsteps=['500','1000','1500','2000']
strategys=["GRAM1","GRAM2","AMSTENCIL"]
GRAM1=1
GRAM2=1
AMSTENCIL=1
monitors=[1,2]
ratios=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
print(ratios)
datatypes=["double"]
if not os.path.exists("./error/motivation"):
    os.makedirs("./error/motivation")
#amstencil
if AMSTENCIL:
    for monitor in monitors:
        for datatype in datatypes:
            # amstencil
            avg_amstencil_error=pd.DataFrame(columns=["timestep","ratio","avg_error"])
            tmp=ratios.copy()
            for i in range(len(timsteps)-1):
                tmp.extend(ratios)
            avg_amstencil_error["ratio"]=tmp
            tmp=[]
            for timestep in timsteps:
                for ratio in ratios:
                    tmp.append(timestep)
            avg_amstencil_error["timestep"]=tmp
            sum_amstencil=None
            for para_index in range(1,6):
                error_thispara_list=[]
                for timestep in timsteps:
                    data_init=pd.read_csv("./result/motivation/"+datatype+"/para"+str(para_index)+"/PURE/"+timestep+"/"+timestep+".csv",index_col=0)
                    for ratio in ratios:
                        data_amstencil=pd.read_csv("./result/motivation/"+datatype+"/para"+str(para_index)+"/AMSTENCIL/monitor"+str(monitor)+"/"+timestep+"/"+str(ratio)+".csv",index_col=0)
                        error_thispara_list.append(np.sum(np.sum(abs(data_init-data_amstencil)))/(512*512))
                # print(error_thispara_list)
                sum_amstencil=pd.concat([sum_amstencil,pd.Series(error_thispara_list)],axis=1)
            avg_amstencil_error["avg_error"]=sum_amstencil.mean(axis=1)
            if monitor==1:
                avg_amstencil_error.to_csv("./error/motivation/AMPStencil-spa.csv",index=None)
            if monitor==2:
                avg_amstencil_error.to_csv("./error/motivation/AMPStencil-tem.csv",index=None)
#gram1
if GRAM1:
    for datatype in datatypes:
        avg_gram1_error=pd.DataFrame(columns=["timestep","ratio","avg_error"])
        tmp=ratios.copy()
        for i in range(len(timsteps)-1):
            tmp.extend(ratios)
        avg_gram1_error["ratio"]=tmp
        tmp=[]
        for timestep in timsteps:
            for ratio in ratios:
                tmp.append(timestep)
        avg_gram1_error["timestep"]=tmp
        sum_gram1=None
        for para_index in range(1,6):
            error_thispara_list_gram1=[]
            for timestep in timsteps:
                data_init=pd.read_csv("./result/motivation/"+datatype+"/para"+str(para_index)+"/PURE/"+timestep+"/"+timestep+".csv",index_col=0)
                for ratio in ratios:
                    data_gram1=pd.read_csv("./result/motivation/"+datatype+"/para"+str(para_index)+"/GRAM1/"+timestep+"/"+str(ratio)+".csv",index_col=0)
                    error_thispara_list_gram1.append(np.sum(np.sum(abs(data_init-data_gram1)))/(512*512))
            sum_gram1=pd.concat([sum_gram1,pd.Series(error_thispara_list_gram1)],axis=1)
        avg_gram1_error["avg_error"]=sum_gram1.mean(axis=1)
        avg_gram1_error.to_csv("./error/motivation/GRAM-clu.csv",index=None)

#gram2
if GRAM2:
    for datatype in datatypes:
        avg_gram2_error=pd.DataFrame(columns=["timestep","ratio","avg_error"])
        tmp=ratios.copy()
        for i in range(len(timsteps)-1):
            tmp.extend(ratios)
        avg_gram2_error["ratio"]=tmp
        tmp=[]
        for timestep in timsteps:
            for ratio in ratios:
                tmp.append(timestep)
        avg_gram2_error["timestep"]=tmp
        sum_gram2=None
        for para_index in range(1,6):
            error_thispara_list_gram2=[]
            for timestep in timsteps:
                data_init=pd.read_csv("./result/motivation/"+datatype+"/para"+str(para_index)+"/PURE/"+timestep+"/"+timestep+".csv",index_col=0)
                for ratio in ratios:
                    data_gram2=pd.read_csv("./result/motivation/"+datatype+"/para"+str(para_index)+"/GRAM2/"+timestep+"/"+str(ratio)+".csv",index_col=0)
                    error_thispara_list_gram2.append(np.sum(np.sum(abs(data_init-data_gram2)))/(512*512))
            sum_gram2=pd.concat([sum_gram2,pd.Series(error_thispara_list_gram2)],axis=1)
        avg_gram2_error["avg_error"]=sum_gram2.mean(axis=1)
        avg_gram2_error.to_csv("./error/motivation/GRAM-sca.csv",index=None)
