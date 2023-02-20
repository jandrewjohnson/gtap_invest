set TABFILE=allshocks

tablo -pgs %TABFILE%
gemsim -cmf %TABFILE%.cmf -p0=%TABFILE% -p1=..\data