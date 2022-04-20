import pandas as pd
import numpy as np
import os
amstecil=1
gram1=1
gram2=1
timestep='500'
thresholds=None
path_error="./error/end2end/"
if not os.path.exists(path_error):
    os.makedirs(path_error)
strategys=["GRAM1","GRAM2","AMSTENCIL"]
ratios=[]
for i in range(1,101):
    ratios.append("%.6f"%(i*0.01))
datatypes=["double","float"]
if amstecil:
    for monitor in [1,2]:
        if monitor==1:
            thresholds=[0.000001,0.00001,0.0001,0.001,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
        if monitor==2:
            thresholds=[1e-20,1e-19,1e-18,1e-17,1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
        for datatype in datatypes:
            avg_amstencil_error=pd.DataFrame(columns=["threshold","avg_error"])
            avg_amstencil_error["threshold"]=thresholds
            sum_amstencil=None
            for para_index in range(1,6):
                data_init=pd.read_csv("./result/end2end/"+datatype+"/pure/para"+str(para_index)+"/"+timestep+".csv",index_col=0)
                error_thispara_list=[]
                for threshold in thresholds:
                    data_amstencil=None
                    if monitor==1:
                        data_amstencil=pd.read_csv("./result/end2end/"+datatype+"/amstencil/monitor"+str(monitor)+"/para"+str(para_index)+"/"+timestep+"_"+"%.6f"%threshold+".csv",index_col=0)
                    if monitor==2:
                        data_amstencil=pd.read_csv("./result/end2end/"+datatype+"/amstencil/monitor"+str(monitor)+"/para"+str(para_index)+"/"+timestep+"_"+str(threshold)+".csv",index_col=0)
                    error_thispara_list.append(np.sum(np.sum(abs(data_init-data_amstencil)))/(512*512))
                sum_amstencil=pd.concat([sum_amstencil,pd.Series(error_thispara_list)],axis=1)
            avg_amstencil_error["avg_error"]=sum_amstencil.mean(axis=1)
            if monitor==1:
                avg_amstencil_error.to_csv(path_error+"AMPStencil-spa_"+datatype+".csv",index=None)
            if monitor==2:
                avg_amstencil_error.to_csv(path_error+"AMPStencil-tem_"+datatype+".csv",index=None)
if gram1:
    for datatype in datatypes:
        avg_gram1_error=pd.DataFrame(columns=["ratio","avg_error"])
        avg_gram1_error["ratio"]=ratios
        sum_gram1=None
        for para_index in range(1,6):
            data_init=pd.read_csv("./result/end2end/"+datatype+"/pure/para"+str(para_index)+"/"+timestep+".csv",index_col=0)
            error_thispara_list_gram1=[]
            for ratio in ratios:
                data_gram1=pd.read_csv("./result/end2end/"+datatype+"/gram1/para"+str(para_index)+"/"+timestep+"_"+ratio+".csv",index_col=0)
                error_thispara_list_gram1.append(np.sum(np.sum(abs(data_init-data_gram1)))/(512*512))
            sum_gram1=pd.concat([sum_gram1,pd.Series(error_thispara_list_gram1)],axis=1)
        avg_gram1_error["avg_error"]=sum_gram1.mean(axis=1)
        avg_gram1_error.to_csv(path_error+"GRAM-clu_"+datatype+".csv",index=None)
if gram2:
    for datatype in datatypes:
        avg_gram2_error=pd.DataFrame(columns=["ratio","avg_error"])
        avg_gram2_error["ratio"]=ratios   
        sum_gram2=None
        for para_index in range(1,6):
            data_init=pd.read_csv("./result/end2end/"+datatype+"/pure/para"+str(para_index)+"/"+timestep+".csv",index_col=0)
            error_thispara_list_gram2=[]
            for ratio in ratios:
                data_gram2=pd.read_csv("./result/end2end/"+datatype+"/gram2/para"+str(para_index)+"/"+timestep+"_"+ratio+".csv",index_col=0)
                error_thispara_list_gram2.append(np.sum(np.sum(abs(data_init-data_gram2)))/(512*512))
            sum_gram2=pd.concat([sum_gram2,pd.Series(error_thispara_list_gram2)],axis=1)
        avg_gram2_error["avg_error"]=sum_gram2.mean(axis=1)
        avg_gram2_error.to_csv(path_error+"/GRAM-sca_"+datatype+".csv",index=None)
