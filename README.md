# AMPStencil-artifact

Code structure

    The directories sintering, graingrowth and snow respectively contain the implementations and evaluations of the three applications in the paper.
    The directories sintreing/paras,graingrowth/paras,snow/paras contain five different parameters corresponding to three application benchmarks.
    The files sintreing/function.h,graingrowth/function.h,snow/function.h implement 3 applications' computation kernels.
    The files of .py implement the running script of evaluations.

Build

    git clone https://github.com/w39614777/w3914777.git
    
Table 3 for Snowflake Crystal Growth:

    Commands to run:
        cd w3914777/snowflake
        python end2end_mix.py
        python end2end_pure.py
        python end2enderror.py
        python get_speedup.py
    Files to store results:
        w3914777/snowflake/error/end2end/
        w3914777/snowflake/speedup/
        
        
Table 3 for Solid-State Sintering:

      Commands to run:(The following 4 commands should be executed serially)
        cd w3914777/sintering
        python end2end_run1.py
        pyrhon end2end_run2.py 
        python get_speedup.py 
        python end2enderror.py 
    Files to store results:
        w3914777/sintering/error/end2end/
        w3914777/sintering/speedup/
        
        
Table 3 for Grain growth:

    Commands to run:
        cd w3914777/graingrowth
        python end2end_mix.py
        python end2end_pure.py
        python end2enderror.py
        python get_speedup.py
    Files to store results:
        w3914777/graingrowth/error/end2end/
        w3914777/graingrowth/speedup/
        
        
Figure 7:

    Commands to run:
        cd w3914777/snowflake
        python motiavtion_mix.py 
        python motivation_pure.py
        python motivationerror.py
    Fils to store results:
        w3914777/snowflake/error/motivation
        
        
Figure 8:

    Commands to run:
        cd w3914777/sintering
        python motivation_run.py 
        python motivationerror.py
    Fils to store results:
        w3914777/sintering/error/motivation
        
        
Figure 9:

    Commands to run:
        cd w3914777/graingrowth
        python motiavtion_mix.py 
        python motivation_pure.py
        python motivationerror.py
    Fils to store results:
         w3914777/graingrowth/error/motivation
         
         
Figure 10:

    Commands to run:
        cd w3914777/snowflake
        python simulation_result.py
    Fils to store results:
        w3914777/snowflake/simulation_result
        
        
Table 4 for Snowflake Crystal Growth:

    Commands to run:
        python time_without_monitorandconversion.py
        python monitor_percent.py
     Fils to store results:
        w3914777/snowflake/monitor:total
        
        
Table 4 for Solid-State Sintering:

    Commands to run:
        python monitor_independent_withoutmonitor.py
        python monitor_independent_withmonitor.py 
        python monitor_percent.py 
     Fils to store results:
        w3914777/graingrowth/monitor:total  
        
        
Table 4 for Grain growth:

    Commands to run:
        python monitor_independent_withoutmonitor.py
        python monitor_independent_withmonitor.py 
        python monitor_percent.py 
     Fils to store results:
        w3914777/graingrowth/monitor:total        
 
