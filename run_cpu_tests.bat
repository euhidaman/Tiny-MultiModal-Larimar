@echo off
REM Windows batch script to run CPU compatibility tests with real BabyLM dataset

echo Starting CPU Compatibility Tests for Tiny-MultiModal-Larimar...
echo This will download the actual BabyLM multimodal dataset (~3GB)

REM Activate virtual environment
call "d:\BabyLM\venv\Scripts\activate.bat"

REM Check if activation worked
if errorlevel 1 (
    echo Failed to activate virtual environment
    echo Please make sure the virtual environment exists at d:\BabyLM\venv\
    pause
    exit /b 1
)

echo Virtual environment activated successfully

REM Install additional dependencies if needed
echo Installing required packages...
pip install requests tqdm > nul 2>&1

echo.
echo =================================================
echo Running CPU Compatibility Test with Real Dataset
echo This will download actual BabyLM data and test compatibility
echo =================================================
python test_cpu_compatibility.py

echo.
echo All tests completed!
pause
