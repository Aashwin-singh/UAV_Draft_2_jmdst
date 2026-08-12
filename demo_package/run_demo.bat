@echo off
REM One-click live demo. Runs preflight first and stops if anything is blocking,
REM so failures surface here rather than in front of an audience.
setlocal
cd /d "%~dp0\.."

set CLIP=%1
if "%CLIP%"=="" set CLIP=demo_package\clips\uavdt_M0101.mp4

call conda activate jmdst
if errorlevel 1 (
  echo Could not activate the 'jmdst' conda env. See demo_package\README.md.
  pause
  exit /b 1
)

python scripts\preflight.py --source "%CLIP%"
if errorlevel 1 (
  echo.
  echo Preflight found a blocking issue - see above. Not launching.
  pause
  exit /b 1
)

echo.
echo Launching demo on %CLIP%  (press q in the window to stop)
python scripts\demo.py --source "%CLIP%" --show
pause
