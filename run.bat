@echo off
setlocal

REM ============================================================================
REM  run.bat — SimpleVox launcher
REM ============================================================================
REM
REM  Usage:
REM      run.bat "path\to\video.mkv"
REM
REM  If no argument is given, processes all video files in input\ folder.
REM
REM  This batch file is a thin wrapper around run.py (Python) which handles
REM  all filenames correctly, including those with special characters like &.
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
if "%~1"=="" (
    echo.
    echo   No input file specified. Processing all videos in input\ folder...
    echo.
    setlocal enabledelayedexpansion
    set "ANY=0"
    for %%E in (mkv mp4 m4v mov avi webm wmv flv) do (
        if exist "input\*.%%E" (
            for %%F in ("input\*.%%E") do (
                echo   -- Processing: %%F
                python run.py "%%F"
                if errorlevel 1 goto :error
                set "ANY=1"
            )
        )
    )
    if "!ANY!"=="0" (
        echo   No video files found in input\.
        goto :error
    )
    endlocal
) else (
    python run.py "%~1"
)

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