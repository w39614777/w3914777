import pandas as pd
import numpy as np
import os
monitor=2
amstecil=1
gram1=0
gram2=0
timestep='500'
path_error="./error/end2end/monitor"+str(monitor)+"/"
if not os.path.exists(path_error):
    os.makedirs(path_error)
thresholds=None
if monitor==1:
    thresholds=[0.000001,0.00001,0.0001,0.001,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
if monitor==2:
    thresholds=[1e-20,1e-19,1e-18,1e-17,1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
strategys=["GRAM1","GRAM2","AMSTENCIL"]
ratios=[]
for i in range(1,101):
    ratios.append(str(round(0.01*i,2)))
datatypes=["double","float"]
if amstecil:
    for datatype in datatypes:
        avg_amstencil_error=pd.DataFrame(columns=["threshold","avg_error"])
        avg_amstencil_error["threshold"]=thresholds
        sum_amstencil=None
        for para_index in range(1,6):
            data_init=pd.read_csv("./result/end2end/"+datatype+"/para"+str(para_index)+"/PURE/"+timestep+".csv",index_col=0)
            error_thispara_list=[]
            for threshold in thresholds:
                data_amstencil=pd.read_csv("./result/end2end/"+datatype+"/para"+str(para_index)+"/amstencil/monitor"+str(monitor)+"/"+str(threshold)+".csv",index_col=0)
                error_thispara_list.append(np.sum(np.sum(abs(data_init-data_amstencil)))/(512*512))
            sum_amstencil=pd.concat([sum_amstencil,pd.Series(error_thispara_list)],axis=1)
        avg_amstencil_error["avg_error"]=sum_amstencil.mean(axis=1)
        avg_amstencil_error.to_csv(path_error+"amstencil_"+datatype+"_error.csv",index=None)
        # pd.concat([avg_amstencil_error["threshold"],sum_amstencil],axis=1).to_csv(path_error+"amstencil_"+datatype+"_error_detail.csv",header=None,index=None)
if gram1:
    for datatype in datatypes:
        avg_gram1_error=pd.DataFrame(columns=["alpha","avg_error"])
        avg_gram1_error["alpha"]=ratios
        sum_gram1=None
        for para_index in range(1,6):
            data_init=pd.read_csv("./result/end2end/"+datatype+"/para"+str(para_index)+"/PURE/"+timestep+".csv",index_col=0)
            error_thispara_list_gram1=[]
            for ratio in ratios:
                data_gram1=pd.read_csv("./result/end2end/"+datatype+"/para"+str(para_index)+"/GRAM1/"+str(ratio)+".csv",index_col=0)
                error_thispara_list_gram1.append(np.sum(np.sum(abs(data_init-data_gram1)))/(512*512))
            sum_gram1=pd.concat([sum_gram1,pd.Series(error_thispara_list_gram1)],axis=1)
        avg_gram1_error["avg_error"]=sum_gram1.mean(axis=1)
        avg_gram1_error.to_csv(path_error+"gram1_"+datatype+"_error.csv",index=None)
        # pd.concat([avg_gram1_error["alpha"],sum_gram1],axis=1).to_csv(path_error+"gram1_"+datatype+"_error_detail.csv",header=None,index=None)
if gram2:
    for datatype in datatypes:
        avg_gram2_error=pd.DataFrame(columns=["alpha","avg_error"])
        avg_gram2_error["alpha"]=ratios   
        sum_gram2=None
        for para_index in range(1,6):
            data_init=pd.read_csv("./result/end2end/"+datatype+"/para"+str(para_index)+"/PURE/"+timestep+".csv",index_col=0)
            error_thispara_list_gram2=[]
            for ratio in ratios:
                data_gram2=pd.read_csv("./result/end2end/"+datatype+"/para"+str(para_index)+"/GRAM2/"+str(ratio)+".csv",index_col=0)
                error_thispara_list_gram2.append(np.sum(np.sum(abs(data_init-data_gram2)))/(512*512))
            sum_gram2=pd.concat([sum_gram2,pd.Series(error_thispara_list_gram2)],axis=1)
        avg_gram2_error["avg_error"]=sum_gram2.mean(axis=1)
        avg_gram2_error.to_csv(path_error+"/gram2_"+datatype+"_error.csv",index=None)
        # pd.concat([avg_gram2_error["alpha"],sum_gram2],axis=1).to_csv(path_error+"gram2_"+datatype+"_error_detail.csv",header=None,index=None)