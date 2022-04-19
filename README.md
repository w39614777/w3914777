# AMPStencil-artifact

Code structure
The directories sintering, graingrowth and snow respectively contain the implementations and evaluations of the three applications in the paper
The directories sintreing/paras,graingrowth/paras,snow/paras contain five different parameters corresponding to three application benchmarks
The files sintreing/function.h,graingrowth/function.h,snow/function.h implement 3 applications' computation kernels
The files of .py implement the running script of evaluations

build


Running
  For sintering
    cd sintering
    
    For motivation:
      Commands to run:(The following 2 commands should be executed serially)
        python motivation_run.py //This command will get the motivation simulation result of AMPStencil,GRAM-clu,GRAM-sca and Pure FP64/FP32.
        python motivationerror.py //This command will get the average absolute error of AMPStencil,GRAM-clu,GRAM-sca relative to Pure FP64/FP32.
      Fils to store results:
        The error of  GRAM-clu(FP64 mixed with FP16) in file sintering/error/motivation/gram1_double.csv
        The error of  GRAM-sca(FP64 mixed with FP16) in file sintering/error/motivation/gram2_double.csv
        The error of  AMPStencil-spa(FP64 mixed with FP16) in file sintering/error/motivation/amstencil_double_monitor1.csv
        The error of  AMPStencil-tem(FP64 mixed with FP16) in file sintering/error/motivation/amstencil_double_monitor2.csv
        The error of  GRAM-clu(FP32 mixed with FP16) in file sintering/error/motivation/gram1_float.csv
        The error of  GRAM-sca(FP32 mixed with FP16) in file sintering/error/motivation/gram2_float.csv
        The error of  AMPStencil-spa(FP32 mixed with FP16) in file sintering/error/motivation/amstencil_float_monitor1.csv
        The error of  AMPStencil-tem(FP32 mixed with FP16) in file sintering/error/motivation/amstencil_float_monitor2.csv

    For end2end:
      Commands to run:(The following 4 commands should be executed serially)
        python end2end_run1.py//This command will get the simulation result of AMPStencil,GRAM-clu,GRAM-sca and Pure FP64/FP32.
        pyrhon end2end_run2.py //This command will get the simulation duration of AMPStencil,GRAM-clu,GRAM-sca and Pure FP64/FP32.
        python get_speedup.py //This command will get the speedup of AMPStencil,GRAM-clu,GRAM-sca relative to Pure FP64/FP32
        python end2enderror.py //This command will get the average absolute error of AMPStencil,GRAM-clu,GRAM-sca relative to Pure FP64/FP32
      Fils to store results:
        The error of  GRAM-clu(FP64 mixed with FP16) in file sintering/error/end2end/gram1_double_error.csv
        The error of  GRAM-sca(FP64 mixed with FP16) in file sintering/error/end2end/gram2_double_error.csv
        The error of  AMPStencil-spa(FP64 mixed with FP16) in file sintering/error/end2end/monitor1/amstencil_double_error.csv
        The error of  AMPStencil-tem(FP64 mixed with FP16) in file sintering/error/end2end/monitor2/amstencil_double_error.csv
        The error of  GRAM-clu(FP32 mixed with FP16) in file sintering/error/end2end/gram1_float_error.csv
        The error of  GRAM-sca(FP32 mixed with FP16) in file sintering/error/end2end/gram2_float_error.csv
        The error of  AMPStencil-spa(FP32 mixed with FP16) in file sintering/error/end2end/monitor1/amstencil_float_error.csv
        The error of  AMPStencil-tem(FP32 mixed with FP16) in file sintering/error/end2end/monitor2/amstencil_float_error.csv
        The speedup of GRAM-clu(FP64 mixed with FP16) in file sintering/speedup/duouble/gram1/gram1.csv
        The speedup of GRAM-sca(FP64 mixed with FP16) in file sintering/speedup/duouble/gram2/gram2.csv
        The speedup of AMPStencil-spa(FP64 mixed with FP16) in file sintering/speedup/duouble/amstencil/monitor1/amstencil.csv
        The speedup of AMPStencil-sca(FP64 mixed with FP16) in file sintering/speedup/duouble/amstencil/monitor2/amstencil.csv      
        The speedup of GRAM-clu(FP32 mixed with FP16) in file sintering/speedup/float/gram1/gram1.csv
        The speedup of GRAM-sca(FP32 mixed with FP16) in file sintering/speedup/float/gram2/gram2.csv
        The speedup of AMPStencil-spa(FP32 mixed with FP16) in file sintering/speedup/float/amstencil/monitor1/amstencil.csv
        The speedup of AMPStencil-sca(FP32 mixed with FP16) in file sintering/speedup/float/amstencil/monitor2/amstencil.csv
        
     For overhead of AMPStencil:
      Commands to run:(The following 2 commands should be executed serially)
        python time_without_monitorandconversion.py//This command will get the AMPStencil simulation duration exclude monitor
        python monitor_percent.py //This command will get the percentage of monitor duration in the total duration
      Fils to store results:
        The monitor percentage of AMPStencil-spa(FP64 mixed with FP16) in file sintering/monitor:total/500/double/monitor1/percentage.csv
        The monitor percentage of AMPStencil-tem(FP64 mixed with FP16) in file sintering/monitor:total/500/double/monitor2/percentage.csv
        The monitor percentage of AMPStencil-spa(FP32 mixed with FP16) in file sintering/monitor:total/500/float/monitor1/percentage.csv
        The monitor percentage of AMPStencil-tem(FP32 mixed with FP16) in file sintering/monitor:total/500/float/monitor2/percentage.csv     
        
        
  For graingrowth 
    cd graingrowth
    
    For motivation:
      Commands to run:(The following 3 commands should be executed serially)
        python motiavtion_mix.py //This command will get the motivation simulation result of AMPStencil,GRAM-clu,GRAM-sca
        python motivation_pure.py//This command will get the motivation simulation result of Pure FP64/FP32
        python motivationerror.py //This command will get the average absolute error of AMPStencil,GRAM-clu,GRAM-sca relative to Pure FP64/FP32
      Fils to store results:
        The error of  GRAM-clu(FP64 mixed with FP16) in file graingrowth/error/motivation/gram1_double.csv
        The error of  GRAM-sca(FP64 mixed with FP16) in file graingrowth/error/motivation/gram2_double.csv
        The error of  AMPStencil-spa(FP64 mixed with FP16) in file graingrowth/error/motivation/amstencil_double_monitor1.csv
        The error of  AMPStencil-tem(FP64 mixed with FP16) in file graingrowth/error/motivation/amstencil_double_monitor2.csv
        The error of  GRAM-clu(FP32 mixed with FP16) in file graingrowth/error/motivation/gram1_float.csv
        The error of  GRAM-sca(FP32 mixed with FP16) in file graingrowth/error/motivation/gram2_float.csv
        The error of  AMPStencil-spa(FP32 mixed with FP16) in file graingrowth/error/motivation/amstencil_float_monitor1.csv
        The error of  AMPStencil-tem(FP32 mixed with FP16) in file graingrowth/error/motivation/amstencil_float_monitor2.csv
        
    For end2end:
      Commands to run:(The following 4 commands should be executed serially)
        python end2end_mix.py //This command will get the simulation result and simulation duration of AMPStencil,GRAM-clu,GRAM-sca
        python end2end_pure.py //This command will get the simualtion result and simulation duaration of Pure FP64/FP32
        python end2enderror.py //This command will get the average absolute error of AMPStencil,GRAM-clu,GRAM-sca relative to Pure FP64/FP32
        python get_speedup.py //This command will get the speedup of AMPStencil,GRAM-clu,GRAM-sca relative to Pure FP64/FP32
      Fils to store results:
        The error of  GRAM-clu(FP64 mixed with FP16) in file graingrowth/error/end2end/gram1_double_error.csv
        The error of  GRAM-sca(FP64 mixed with FP16) in file graingrowth/error/end2end/gram2_double_error.csv
        The error of  AMPStencil-spa(FP64 mixed with FP16) in file graingrowth/error/end2end/monitor1/amstencil_double_error.csv
        The error of  AMPStencil-tem(FP64 mixed with FP16) in file graingrowth/error/end2end/monitor2/amstencil_double_error.csv
        The error of  GRAM-clu(FP32 mixed with FP16) in file graingrowth/error/end2end/gram1_float_error.csv
        The error of  GRAM-sca(FP32 mixed with FP16) in file graingrowth/error/end2end/gram2_float_error.csv
        The error of  AMPStencil-spa(FP32 mixed with FP16) in file graingrowth/error/end2end/monitor1/amstencil_float_error.csv
        The error of  AMPStencil-tem(FP32 mixed with FP16) in file graingrowth/error/end2end/monitor2/amstencil_float_error.csv
        The speedup of GRAM-clu(FP64 mixed with FP16) in file graingrowth/speedup/double/gram1/gram1.csv
        The speedup of GRAM-sca(FP64 mixed with FP16) in file graingrowth/speedup/double/gram2/gram2.csv
        The speedup of AMPStencil-spa(FP64 mixed with FP16) in file graingrowth/speedup/double/amstencil/monitor1/amstencil.csv
        The speedup of AMPStencil-sca(FP64 mixed with FP16) in file graingrowth/speedup/double/amstencil/monitor2/amstencil.csv
        The speedup of GRAM-clu(FP32 mixed with FP16) in file graingrowth/speedup/float/gram1/gram1.csv
        The speedup of GRAM-sca(FP32 mixed with FP16) in file graingrowth/speedup/float/gram2/gram2.csv
        The speedup of AMPStencil-spa(FP32 mixed with FP16) in file graingrowth/speedup/float/amstencil/monitor1/amstencil.csv
        The speedup of AMPStencil-sca(FP32 mixed with FP16) in file graingrowth/speedup/float/amstencil/monitor2/amstencil.csv
        
     For overhead of AMPStencil:
      Commands to run:(The following 3 commands should be executed serially)
        python monitor_independent_withoutmonitor.py//This command will get the AMPStencil simulation duration exclude monitor
        python monitor_independent_withmonitor.py // This command will get the AMPStencil simulation duration include monitor
        python monitor_percent.py //This command will get the percentage of monitor duration in the total duration
      Fils to store results:
        The monitor percentage of AMPStencil-spa(FP64 mixed with FP16) in file graingrowth/monitor:total/500/double/monitor1/percentage.csv
        The monitor percentage of AMPStencil-tem(FP64 mixed with FP16) in file graingrowth/monitor:total/500/double/monitor2/percentage.csv
        The monitor percentage of AMPStencil-spa(FP32 mixed with FP16) in file graingrowth/monitor:total/500/float/monitor1/percentage.csv
        The monitor percentage of AMPStencil-tem(FP32 mixed with FP16) in file graingrowth/monitor:total/500/float/monitor2/percentage.csv
        
        
  For snow:  
    The running commands and files to store results is similar with graingrowth.      
 
