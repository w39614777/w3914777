# AMPStencil-artifact

Code structure

    ./sintering, ./graingrowth and ./snow: source code of simulation applicaions, including implementation of baseline, GRAM and AMPStencil
    ./app name/paras: benchmarks used in evaluation.

Build

    git clone https://github.com/w39614777/w3914777.git
    
    
Running    
    
    nvidia-smi -lgc 1410 -i GPU_Number
    export CUDA_VISIBLE_DEVICES=GPU_Number
    
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
        python end2end_run2.py 
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
        w3914777/snowflake/error/motivation/
        
        
Figure 8:

    Commands to run:
        cd w3914777/sintering
        python motivation_run.py 
        python motivationerror.py
    Fils to store results:
        w3914777/sintering/error/motivation/
        
        
Figure 9:

    Commands to run:
        cd w3914777/graingrowth
        python motiavtion_mix.py 
        python motivation_pure.py
        python motivationerror.py
    Fils to store results:
         w3914777/graingrowth/error/motivation/
         
         
Figure 10:

    Commands to run:
        cd w3914777/snowflake
        python simulation_result.py
    Fils to store results:
        w3914777/snowflake/simulation_result/
        
        
Table 4 for Snowflake Crystal Growth:

    Commands to run:
        cd w3914777/snowflake
        python monitor_independent_withmonitor.py
        python monitor_independent_withoutmonitor.py
        python monitor_percent.py
     Fils to store results:
        w3914777/snowflake/monitor2total/
        
        
Table 4 for Solid-State Sintering:

    Commands to run: (This experiment should to be behind experiment Table 3 for Solid-State Sintering)
        cd w3914777/sintering
        python time_without_monitorandconversion.py
        python monitor_percent.py 
     Fils to store results:
        w3914777/graingrowth/monitor2total/  
        
        
Table 4 for Grain growth:

    Commands to run:
        cd w3914777/graingrowth
        python monitor_independent_withoutmonitor.py
        python monitor_independent_withmonitor.py 
        python monitor_percent.py 
     Fils to store results:
        w3914777/graingrowth/monitor2total/       
 
