<#
.SYNOPSIS
    Helper script for Windows users (PowerShell) to replace GNU Make targets.
    Usage examples:
        .\setup.ps1 setup
        .\setup.ps1 serve
        .\setup.ps1 test
        .\setup.ps1 train
        .\setup.ps1 calibrate
        .\setup.ps1 export-onnx
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("setup", "serve", "test", "train", "calibrate", "export-onnx", "fmt", "lint")]
    [string]$Task = "setup"
)

function Run-Command {
    param([string]$Cmd)
    Write-Host ">> $Cmd" -ForegroundColor Cyan
    Invoke-Expression $Cmd
}

switch ($Task) {
    "setup" {
        Run-Command "python -m pip install --upgrade pip"
        Run-Command "python -m pip install -r requirements.txt"
    }
    "serve" {
        Run-Command "uvicorn run_server:app --reload --port 8000"
    }
    "test" {
        Run-Command "pytest -q"
    }
    "train" {
        Run-Command "python scripts/train_pa.py"
    }
    "calibrate" {
        Run-Command "python scripts/calibrate_pa.py"
    }
    "export-onnx" {
        Run-Command "python scripts/export_onnx.py"
    }
    "fmt" {
        Run-Command "python -m pip install ruff black"
        Run-Command "ruff check --fix ."
        Run-Command "black ."
    }
    "lint" {
        Run-Command "ruff check ."
    }
}
