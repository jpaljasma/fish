@echo off
setlocal
rem Restore SDK tool lookup when launched from omp/MSYS. Keep changes local.
set "PATH=%SystemRoot%\System32;%SystemRoot%;%PATH%"
set "vswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%I in (`"%vswhere%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "vsroot=%%I"
if not defined vsroot (
  echo Install Visual Studio Build Tools with the x64 C++ toolchain. 1>&2
  exit /b 1
)
call "%vsroot%\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"
if exist "%ProgramFiles%\CMake\bin\cmake.exe" set "PATH=%ProgramFiles%\CMake\bin;%PATH%"
cd /d "%~dp0.." || exit /b 1
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release || exit /b 1
cmake --build build || exit /b 1
ctest --test-dir build --output-on-failure
exit /b %errorlevel%
