 ::======================================================================
:: Step 1: Define shock and data directories
::======================================================================
set SHKd=.\shocks
set Datd=..\data

::======================================================================
:: Step 2: Define TAB and cmf files 
::======================================================================
set TAB01=ALLShocks

::======================================================================
:: Step 3: Create shocks 
::======================================================================
cd %SHKd%

:: Step 3A: Convert Text to HAR 
call Conv_txt2hars.bat 

:: Step 3B: Create shocks file 
 tablo -pgs %TAB01%
  if errorlevel 1 goto error

 gemsim -cmf %TAB01%.cmf -p0=%TAB01% -p1=%Datd%
  if errorlevel 1 goto error

cd..

echo Job done OK    
dir/od *.har 
goto endOK   

:error     
dir/od *.log
echo PROBLEM !!!! examine most recent Log
exit /b 1         

:endOK              
