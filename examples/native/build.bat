@echo off
rem Build vecmath.c into vecmath.dll on Windows 10/11
rem Uses MSVC (run from "x64 Native Tools Command Prompt") or MinGW-w64 gcc
setlocal
cd /d "%~dp0"
where cl >nul 2>nul
if %ERRORLEVEL%==0 (
    cl /nologo /O2 /arch:AVX2 /LD vecmath.c /Fe:vecmath.dll
    goto done
)
where gcc >nul 2>nul
if %ERRORLEVEL%==0 (
    gcc -O3 -march=x86-64-v3 -shared vecmath.c -o vecmath.dll
    goto done
)
echo No C compiler found. Install Visual Studio Build Tools or MinGW-w64 first.
exit /b 1
:done
echo built %CD%\vecmath.dll
endlocal
