@echo off
setlocal

pushd %~dp0

if exist venv\Scripts\activate (
    call venv\Scripts\activate
)

python -m streamlit run app.py

popd
endlocal
pause
