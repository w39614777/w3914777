import pandas as pd
import numpy as np
import os
if not os.path.exists("./speedup/"):
    os.makedirs("./speedup/")
for datatype in ['float','double']:
    inittime=None
    for para_index in range(1,6):
        time=pd.read_csv("./time/"+datatype+'/para'+str(para_index)+"/PURE/time.csv",header=None).mean(axis=1)
        inittime=pd.concat([inittime,pd.Series(time)],axis=1)
    inittime=float(inittime.mean(axis=1))
    # gram1
    gram1time=None
    for para_index in range(1,6):
        time=pd.read_csv("./time/"+datatype+"/para"+str(para_index)+"/GRAM1/time.csv",header=None,index_col=0).mean(axis=1)
        gram1time=pd.concat([gram1time,pd.Series(time)],axis=1)
    gram1time=gram1time.mean(axis=1)
    gram1speedup=(inittime/gram1time)
    gram1speedup=pd.DataFrame({'ratio':gram1speedup.index,'speedup':gram1speedup.values})
    gram1speedup.to_csv("./speedup/GRAM-clu_"+datatype+".csv",index=None)
    # gram2
    gram2time=None
    for para_index in range(1,6):
        time=pd.read_csv("./time/"+datatype+"/para"+str(para_index)+"/GRAM2/time.csv",header=None,index_col=0).mean(axis=1)
        gram2time=pd.concat([gram2time,pd.Series(time)],axis=1)
    gram2time=gram2time.mean(axis=1)
    gram2speedup=(inittime/gram2time)
    gram2speedup=pd.DataFrame({'ratio':gram2speedup.index,'speedup':gram2speedup.values})
    gram2speedup.to_csv("./speedup/GRAM-sca_"+datatype+".csv",index=None)
    # amstencil
    for monitor in range(1,3):
        amstenciltime=None
        for para_index in range(1,6):
            time=pd.read_csv("./time/"+datatype+"/para"+str(para_index)+"/amstencil/monitor"+str(monitor)+"/time.csv",header=None,index_col=0).mean(axis=1)
            amstenciltime=pd.concat([amstenciltime,pd.Series(time)],axis=1)
        amstenciltime=amstenciltime.mean(axis=1)
        amstencilspeedup=(inittime/amstenciltime)
        amstencilspeedup=pd.DataFrame({'threshold':amstencilspeedup.index,'speedup':amstencilspeedup.values})
        if monitor==1:

            amstencilspeedup.to_csv("./speedup/AMPStencil-spa_"+datatype+".csv",index=None)
        if monitor==2:
            amstencilspeedup.to_csv("./speedup/AMPStencil-tem_"+datatype+".csv",index=None)
