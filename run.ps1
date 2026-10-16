# One-command launcher for The Money Leak
# Usage:  powershell -ExecutionPolicy Bypass -File .\run.ps1
$ErrorActionPreference = "Stop"
python -m pip install -r requirements.txt
python -m streamlit run app.py
