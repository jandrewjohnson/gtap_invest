
::======================================================================
:: Step 1: Define Directories 
::======================================================================
set MODd=..\mod
set OUTd=..\out
set DATd=..\data
set SHKd=..\shocks
set WELd=..\decomp
set CMFd=.\cmf

set Model=%MODd%\gtapaez
::======================================================================
:: Step 2: Define cmf files 
::======================================================================
set sim00=Test
set sim01=2014_21_BAU
set sim02=2021_30_BAU_noES
set sim03=2021_30_BAU_allES
set sim04=2021_30_PESGC_allES
set sim05=2021_30_PESLC_allES
set sim06=2021_30_SR_Land_allES
set sim07=2021_30_SR_RnD_20p_allES
set sim08=2021_30_SR_RnD_20p_PESGC_allES
set sim09=2021_30_BAU_rigid_allES
set sim10=2021_30_BAU_allES_allCol

::======================================================================
:: Step 4: Run sims 
::======================================================================
cd %CMFd%
goto skip
 %MODEL% -cmf %sim00%.cmf -p1=%DATd% -p2=%OUTd%
     if errorlevel 1 goto error

 %MODEL% -cmf %sim01%.cmf -p1=%DATd% -p2=%OUTd% -p3=%SHKd%
     if errorlevel 1 goto error

 %MODEL% -cmf %sim02%.cmf -p1=%DATd% -p2=%OUTd% -p3=%SHKd%
     if errorlevel 1 goto error

 %MODEL% -cmf %sim03%.cmf -p1=%DATd% -p2=%OUTd% -p3=%SHKd%
     if errorlevel 1 goto error

 %MODEL% -cmf %sim04%.cmf -p1=%DATd% -p2=%OUTd% -p3=%SHKd%
     if errorlevel 1 goto error

 %MODEL% -cmf %sim05%.cmf -p1=%DATd% -p2=%OUTd% -p3=%SHKd%
     if errorlevel 1 goto error

 %MODEL% -cmf %sim06%.cmf -p1=%DATd% -p2=%OUTd% -p3=%SHKd%
     if errorlevel 1 goto error

 %MODEL% -cmf %sim07%.cmf -p1=%DATd% -p2=%OUTd% -p3=%SHKd%
     if errorlevel 1 goto error

 %MODEL% -cmf %sim08%.cmf -p1=%DATd% -p2=%OUTd% -p3=%SHKd%
     if errorlevel 1 goto error

::--------------
:: RIGID scenario
::--------------
 %MODEL% -cmf %sim09%.cmf -p1=%DATd% -p2=%OUTd% -p3=%SHKd%
     if errorlevel 1 goto error
:skip
::----------------------
:: All collapse scenario
::----------------------
 %MODEL% -cmf %sim10%.cmf -p1=%DATd% -p2=%OUTd% -p3=%SHKd%
     if errorlevel 1 goto error

cd..

echo Job done OK    
dir/od *.sl4 
goto endOK   

:error     
dir/od *.log
echo PROBLEM !!!! examine most recent Log
exit /b 1         

:endOK              
