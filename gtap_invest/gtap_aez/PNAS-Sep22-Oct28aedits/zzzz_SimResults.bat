
::  =================================
::  File Setting: May require changes  
::  =================================

::  -------------------------------
:: Step 1: Project name and folders
::  -------------------------------
SET PATHDIR=.\
SET RESDIR=results

SET DATDIR=..\..\data
SET SIMDIR=out

SET RESOUT=..\res
SET RESWRK=..\wrk
SET WELDIR=..\..\decomp


::  -------------------------------
::  Step 2: Define simulation names
::  -------------------------------
 
SET SIM01=2014_21_BAU
SET SIM02=2021_30_BAU_noES
SET SIM03=2021_30_BAU_allES
SET SIM04=2021_30_PESGC_allES
SET SIM05=2021_30_PESLC_allES
SET SIM06=2021_30_SR_Land_allES
SET SIM07=2021_30_SR_RnD_20p_allES
SET SIM08=2021_30_SR_RnD_20p_PESGC_allES
SET sim09=2021_30_BAU_rigid_allES
SET SIM03=2021_30_BAU_allES_allCol

::  ----------------------------
::  DO NOT EDIT BEYOND THIS LINE
::  ----------------------------

SET MAPRES=%RESDIR%\results.map

::  --------------------------
::  Step 3: Copy Results files
::  --------------------------
echo converting results. . .

sltoht -map=%MAPRES% %SIMDIR%\%Sim01% %RESDIR%\in\%Sim01%.sol
     if errorlevel 1 goto error

sltoht -map=%MAPRES% %SIMDIR%\%Sim02% %RESDIR%\in\%Sim02%.sol
     if errorlevel 1 goto error

sltoht -map=%MAPRES% %SIMDIR%\%Sim03% %RESDIR%\in\%Sim03%.sol
     if errorlevel 1 goto error

sltoht -map=%MAPRES% %SIMDIR%\%Sim04% %RESDIR%\in\%Sim04%.sol
     if errorlevel 1 goto error

sltoht -map=%MAPRES% %SIMDIR%\%Sim05% %RESDIR%\in\%Sim05%.sol
     if errorlevel 1 goto error

sltoht -map=%MAPRES% %SIMDIR%\%Sim06% %RESDIR%\in\%Sim06%.sol
     if errorlevel 1 goto error

sltoht -map=%MAPRES% %SIMDIR%\%Sim07% %RESDIR%\in\%Sim07%.sol
     if errorlevel 1 goto error

sltoht -map=%MAPRES% %SIMDIR%\%Sim08% %RESDIR%\in\%Sim08%.sol
     if errorlevel 1 goto error

sltoht -map=%MAPRES% %SIMDIR%\%Sim09% %RESDIR%\in\%Sim09%.sol
     if errorlevel 1 goto error

sltoht -map=%MAPRES% %SIMDIR%\%Sim10% %RESDIR%\in\%Sim10%.sol
     if errorlevel 1 goto error
::  -----------------------
::  Step 4: Combine results
::  -----------------------

echo combine results. . .
cd %RESdir%\src

 cmbhar -sti cmbres.sti

cd../..

::  -----------------------
::  Step 5: Results summary
::  -----------------------

echo create results summary file. . .

cd %RESDIR%\src

tablo -pgs Allres
     if errorlevel 1 goto error

gemsim -cmf Allres.cmf -p1=%DATdir% -p2=%RESWRK% -p3=.. -p4=%WELdir% -p5=%RESout%
     if errorlevel 1 goto error

cd../..


::  -----------------------
::  Step 6: Results in csv
::  -----------------------
cd %RESDIR%\res

::  6A: Welfare results  
::  ---------------------------------------
:: 6A.1 EV by  region
 har2csv allres.har EVR.csv EVR
:: 6A.2 EV by aggregate region
 har2csv allres.har EVAR.csv EVAR
:: 6A.3 EV decomposition by aggregate region
 har2csv allres.har EVAC.csv EVAC

::  6B: GDP results 
::  ---------------------------------------
:: 6B.1 GDP by region
 har2csv allres.har GDPR.csv GDPR
:: 6B.2 GDP expenditure side decomposition by region
 har2csv allres.har GERC.csv GERC
:: 6B.3 GDP income side decomposition by region
 har2csv allres.har GIRC.csv GIRC

:: 6B.4 GDP by aggregate region
 har2csv allres.har GDAR.csv GDAR
:: 6B.5 GDP expenditure side decomposition by aggregate region
 har2csv allres.har GEAC.csv GEAC
:: 6B.6 GDP income side decomposition by aggregate region
 har2csv allres.har GIAC.csv GIAC

::  6D: Regional Production results 
::  -------------------------------
:: 6D.1 Full dimension 
 har2csv allres.har ract.csv ract
:: 6D.2 Aggregated results
 har2csv allres.har raca.csv raca

::  6E: Regional Endowment results 
::  -------------------------------
:: 6E.1 Full dimension 
 har2csv allres.har ends.csv ends
:: 6E.2 Aggregated results
 har2csv allres.har enda.csv enda

::  6F: Factor demand results 
::  -------------------------------
:: 6F.1 Full dimension 
 har2csv allres.har facs.csv facs
:: 6F.1 Aggregated dimension 
 har2csv allres.har faca.csv faca

::  6G: Trade results 
::  -------------------------------
:: 6G.1 Full dimension 
 har2csv allres.har trad.csv trad
:: 6G.1 Aggregated dimension 
 har2csv allres.har trda.csv trda


cd../..

::  =================
::  Terminal messages
::  =================
echo Results processing completed !
goto endOK          

:error           
echo PROBLEM !!!! examine most recent Log
exit /b 1            
:endOK     

::====
:: END
::====
