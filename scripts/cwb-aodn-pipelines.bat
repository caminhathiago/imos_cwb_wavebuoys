echo off

set arg1=%1

if "%arg1%"=="" (
    echo [ERROR] Please pass a valid script name
    echo usage: cwb-aodn-pipelines.bat [python_script_name]
    exit b/ 1
)

REM Set up virtual environment Python path
set PYTHON_EXEC=C:\Users\00116827\venvs\.cwb_aodn_development\Scripts\python.exe
set INCOMING_PATH=C:\Users\00116827\cwb\wavebuoy_aodn\incoming_path

REM Conditional execution based on the passed argument
if "%arg1%"=="cwb-netcdf-aodn" (
    echo [STARTED] %DATE% %TIME%
    %PYTHON_EXEC% C:\Users\00116827\cwb\wavebuoy_aodn\scripts\cwb-netcdf-aodn.py ^
    -o %INCOMING_PATH% ^
    -i %INCOMING_PATH% ^
    -lh 1
    echo [FINISHED] %DATE% %TIME%

) else if "%arg1%"=="cwb-spotter-spectral-netcdf" (
    echo [STARTED] %DATE% %TIME%
    %PYTHON_EXEC% C:\Users\00116827\cwb\wavebuoy_aodn\scripts\cwb-spotter-spectral-netcdf.py ^
    -o %INCOMING_PATH% ^
    -i %INCOMING_PATH% ^
    -w 72 ^
    -wu hours
    echo [FINISHED] %DATE% %TIME%

) else if "%arg1%"=="cwb-spotter-bulk-netcdf" (
    echo [STARTED] %DATE% %TIME%
    %PYTHON_EXEC% C:\Users\00116827\cwb\wavebuoy_aodn\scripts\cwb-spotter-bulk-netcdf.py ^
    -o %INCOMING_PATH% ^
    -i %INCOMING_PATH% ^
    -w 24 ^
    -wu hours
    echo [FINISHED] %DATE% %TIME%

) else (
    echo [ERROR] Invalid script name provided.
    exit /b 1
)