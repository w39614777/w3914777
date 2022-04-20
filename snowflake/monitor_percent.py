import numpy as np 
import pandas as pd
import  os
if not os.path.exists("./monitor2total/"):
    os.makedirs("./monitor2total/")
datatypes=["float","double"]
monitors=['1','2']
for datatype in datatypes:
    for monitor in monitors:
        total_time=None
        monitor_time=None
        for paraindex in range(1,6):
            total_time_thispara=pd.read_csv("./time/"+datatype+"/para"+str(paraindex)+'/with_M'+monitor+'_monitor/time.csv',header=None,index_col=0).mean(axis=1)
            monitor_time_thispara=pd.read_csv("./time/"+datatype+"/para"+str(paraindex)+'/without_M'+monitor+'_monitor/time.csv',header=None,index_col=0).mean(axis=1)
            total_time=pd.concat([total_time,pd.Series(total_time_thispara)],axis=1)
            monitor_time=pd.concat([monitor_time,pd.Series(monitor_time_thispara)],axis=1)
        total_time=total_time.mean(axis=1)
        monitor_time=monitor_time.mean(axis=1)
        percrntage=(total_time-monitor_time)/total_time*100
        percrntage=pd.DataFrame({'threshold':percrntage.index,'monitor_percentage':percrntage.values})
        if monitor=='1':
            percrntage.to_csv("./monitor2total/AMPStencil-spa_"+datatype+".csv",index=None)
        if monitor=='2':
            percrntage.to_csv("./monitor2total/AMPStencil-tem_"+datatype+".csv",index=None)