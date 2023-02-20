
::======================================================================
:: Step 1: Define Directories 
::======================================================================
set SOLd=.\out
set DATd=.\data
set WELd=.\decomp
set RMAP=%WELd%\REGMAP
:: set RMAP2=%WELd%\REGMAP_LowMidHi
set RMAP2=%WELd%\REGMAP_SSALowMidHi

::======================================================================
:: Step 2: Define cmf files 
::======================================================================
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
:: Step 3: Generate welfare SOL files 
::======================================================================

sltoht -map=%WELd%\decomp.map %SOLd%\%Sim01%.sl4 %SOLd%\%Sim01%.sol
     if errorlevel 1 goto error

sltoht -map=%WELd%\decomp.map %SOLd%\%Sim02%.sl4 %SOLd%\%Sim02%.sol
     if errorlevel 1 goto error

sltoht -map=%WELd%\decomp.map %SOLd%\%Sim03%.sl4 %SOLd%\%Sim03%.sol
     if errorlevel 1 goto error

sltoht -map=%WELd%\decomp.map %SOLd%\%Sim04%.sl4 %SOLd%\%Sim04%.sol
     if errorlevel 1 goto error

sltoht -map=%WELd%\decomp.map %SOLd%\%Sim05%.sl4 %SOLd%\%Sim05%.sol
     if errorlevel 1 goto error

sltoht -map=%WELd%\decomp.map %SOLd%\%Sim06%.sl4 %SOLd%\%Sim06%.sol
     if errorlevel 1 goto error

sltoht -map=%WELd%\decomp.map %SOLd%\%Sim07%.sl4 %SOLd%\%Sim07%.sol
     if errorlevel 1 goto error

sltoht -map=%WELd%\decomp.map %SOLd%\%Sim08%.sl4 %SOLd%\%Sim08%.sol
     if errorlevel 1 goto error

::---------------
:: RIGID scenario
::---------------
sltoht -map=%WELd%\decomp.map %SOLd%\%Sim09%.sl4 %SOLd%\%Sim09%.sol
     if errorlevel 1 goto error

::------------------
:: collapse scenario
::------------------
sltoht -map=%WELd%\decomp.map %SOLd%\%Sim10%.sl4 %SOLd%\%Sim10%.sol
     if errorlevel 1 goto error

::======================================================================
:: Step 3: Generate welfare decomp files 
::======================================================================

:: Use basedata
%WELd%\decomp -cmf %WELd%\decomp.cmf -p1=%DATd% -p2=%SOLd%\%Sim01% -p3=%DATd%\basedata.har          -p4=%SOLd%\%Sim01% -p5=%WELd%\%Sim01%-decomp -p6=%RMAP%
     if errorlevel 1 goto error

%WELd%\decomp -cmf %WELd%\decomp.cmf -p1=%DATd% -p2=%SOLd%\%Sim02% -p3=%SOLd%\2014_21_BAU.upd       -p4=%SOLd%\%Sim02% -p5=%WELd%\%Sim02%-decomp -p6=%RMAP%
     if errorlevel 1 goto error

%WELd%\decomp -cmf %WELd%\decomp.cmf -p1=%DATd% -p2=%SOLd%\%Sim03% -p3=%SOLd%\2021_30_BAU_noES.upd  -p4=%SOLd%\%Sim03% -p5=%WELd%\%Sim03%-decomp -p6=%RMAP%
     if errorlevel 1 goto error

%WELd%\decomp -cmf %WELd%\decomp.cmf -p1=%DATd% -p2=%SOLd%\%Sim04% -p3=%SOLd%\2021_30_BAU_allES.upd -p4=%SOLd%\%Sim04% -p5=%WELd%\%Sim04%-decomp -p6=%RMAP%
     if errorlevel 1 goto error

%WELd%\decomp -cmf %WELd%\decomp.cmf -p1=%DATd% -p2=%SOLd%\%Sim05% -p3=%SOLd%\2021_30_BAU_allES.upd -p4=%SOLd%\%Sim05% -p5=%WELd%\%Sim05%-decomp -p6=%RMAP%
     if errorlevel 1 goto error

%WELd%\decomp -cmf %WELd%\decomp.cmf -p1=%DATd% -p2=%SOLd%\%Sim06% -p3=%SOLd%\2021_30_BAU_allES.upd -p4=%SOLd%\%Sim06% -p5=%WELd%\%Sim06%-decomp -p6=%RMAP%
     if errorlevel 1 goto error

%WELd%\decomp -cmf %WELd%\decomp.cmf -p1=%DATd% -p2=%SOLd%\%Sim07% -p3=%SOLd%\2021_30_BAU_allES.upd -p4=%SOLd%\%Sim07% -p5=%WELd%\%Sim07%-decomp -p6=%RMAP%
     if errorlevel 1 goto error

%WELd%\decomp -cmf %WELd%\decomp.cmf -p1=%DATd% -p2=%SOLd%\%Sim08% -p3=%SOLd%\2021_30_BAU_allES.upd -p4=%SOLd%\%Sim08% -p5=%WELd%\%Sim08%-decomp -p6=%RMAP%
     if errorlevel 1 goto error

::--------------
::  RIGID scenario
::--------------
%WELd%\decomp -cmf %WELd%\decomp.cmf -p1=%DATd% -p2=%SOLd%\%Sim09% -p3=%SOLd%\2021_30_BAU_noES.upd  -p4=%SOLd%\%Sim09% -p5=%WELd%\%Sim09%-decomp -p6=%RMAP%
     if errorlevel 1 goto error

::------------------
:: collapse scenario
::------------------
%WELd%\decomp -cmf %WELd%\decomp.cmf -p1=%DATd% -p2=%SOLd%\%Sim10% -p3=%SOLd%\2021_30_BAU_noES.upd  -p4=%SOLd%\%Sim10% -p5=%WELd%\%Sim10%-decomp -p6=%RMAP%
     if errorlevel 1 goto error

::======================================================================
:: Step 4: Welfare decomp with region aggregated by income level 
::======================================================================

:: Delete aggregate welfare decomp file
del %WELd%\agg-%Sim01%-decomp.har
del %WELd%\agg-%Sim02%-decomp.har
del %WELd%\agg-%Sim03%-decomp.har
del %WELd%\agg-%Sim04%-decomp.har
del %WELd%\agg-%Sim05%-decomp.har
del %WELd%\agg-%Sim06%-decomp.har
del %WELd%\agg-%Sim07%-decomp.har
del %WELd%\agg-%Sim08%-decomp.har
del %WELd%\agg-%Sim09%-decomp.har
del %WELd%\agg-%Sim10%-decomp.har

agghar %WELd%\%Sim01%-decomp.har %WELd%\agg-%Sim01%-decomp.har %RMAP2%.har -PM
     if errorlevel 1 goto error

agghar %WELd%\%Sim02%-decomp.har %WELd%\agg-%Sim02%-decomp.har %RMAP2%.har -PM
     if errorlevel 1 goto error

agghar %WELd%\%Sim03%-decomp.har %WELd%\agg-%Sim03%-decomp.har %RMAP2%.har -PM
     if errorlevel 1 goto error

agghar %WELd%\%Sim04%-decomp.har %WELd%\agg-%Sim04%-decomp.har %RMAP2%.har -PM
     if errorlevel 1 goto error

agghar %WELd%\%Sim05%-decomp.har %WELd%\agg-%Sim05%-decomp.har %RMAP2%.har -PM
     if errorlevel 1 goto error

agghar %WELd%\%Sim06%-decomp.har %WELd%\agg-%Sim06%-decomp.har %RMAP2%.har -PM
     if errorlevel 1 goto error

agghar %WELd%\%Sim07%-decomp.har %WELd%\agg-%Sim07%-decomp.har %RMAP2%.har -PM
     if errorlevel 1 goto error

agghar %WELd%\%Sim08%-decomp.har %WELd%\agg-%Sim08%-decomp.har %RMAP2%.har -PM
     if errorlevel 1 goto error

::--------------
  RIGID scenario
::--------------
agghar %WELd%\%Sim09%-decomp.har %WELd%\agg-%Sim09%-decomp.har %RMAP2%.har -PM
     if errorlevel 1 goto error

::------------------
:: collapse scenario
::------------------
agghar %WELd%\%Sim10%-decomp.har %WELd%\agg-%Sim10%-decomp.har %RMAP2%.har -PM
     if errorlevel 1 goto error

::======================================================================
:: Step 5: Welfare decomp summary for all sims 
::======================================================================
cd %WELd%
cmbhar -sti welcmbres.sti

cd.. 

del *.bak

echo Job done OK    
dir/od *-decomp.har 
goto endOK   

:error     
dir/od *.log
echo PROBLEM !!!! examine most recent Log
exit /b 1         

