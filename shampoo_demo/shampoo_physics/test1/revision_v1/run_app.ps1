param([string]$Python = "python")
Set-Location -LiteralPath $PSScriptRoot
& $Python -m streamlit run app.py --server.address 127.0.0.1 --server.headless true
