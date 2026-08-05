@echo off
setlocal

REM ============================================================================
REM  run.bat — SimpleVox launcher
REM ============================================================================
REM
REM  Usage:
REM      run.bat                         (process input\ recursively)
REM      run.bat "path\to\video.mkv"     (single file)
REM      run.bat "path\to\folder"        (whole folder, recursive)
REM      run.bat "input\movie.mkv" --voice en-US-DavisNeural
REM
REM  This batch file is a thin wrapper around run.py (Python), which handles
REM  ALL of the following correctly:
REM    - filenames with spaces, &, !, %, and other special characters
REM    - recursive folder discovery (files in subfolders of input\)
REM    - mirroring the input folder structure into output\
REM
REM  Prerequisites:
REM      1.  Python 3.10+ on PATH
REM      2.  ffmpeg on PATH  (winget install Gyan.FFmpeg)
REM      3.  pip install -r requirements.txt
REM
REM ============================================================================

REM --- Switch to the directory containing this batch file ---
cd /d "%~dp0"

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.10+.
    goto :error
)

REM --- Check ffmpeg ---
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ffmpeg not found on PATH.
    echo         Install: winget install Gyan.FFmpeg
    echo         Then open a NEW terminal and re-run.
    goto :error
)

REM --- Run the pipeline ---
REM  Deliberately do NOT use `for` loops or delayed expansion here: CMD's
REM  `!` handling mangles filenames like "American Dad! S17E11.mkv".
REM  run.py does all discovery and handles special characters safely.
python run.py %*

if errorlevel 1 goto :error

echo.
echo ==============================================================================
echo   [SUCCESS] PIPELINE COMPLETE
echo   Check the output\ folder for results.
echo ==============================================================================
endlocal
exit /b 0

:error
echo.
echo ==============================================================================
echo   [FAILED] PIPELINE STOPPED DUE TO AN ERROR
echo ==============================================================================
endlocal
exit /b 1