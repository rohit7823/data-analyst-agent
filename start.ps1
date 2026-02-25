# AI Agent - One-Click Start
# Starts both backend (port 3000) and frontend (port 3001)

$Host.UI.RawUI.WindowTitle = "AI Agent Launcher"

Write-Host ""
Write-Host "  ======================================" -ForegroundColor Cyan
Write-Host "       AI Data Analyst Agent            " -ForegroundColor Cyan
Write-Host "  ======================================" -ForegroundColor Cyan
Write-Host ""

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check venv exists
$venvActivate = Join-Path $ROOT ".venv\Scripts\Activate.ps1"
if (-Not (Test-Path $venvActivate)) {
    Write-Host "  [ERROR] .venv not found." -ForegroundColor Red
    Write-Host "  Run: python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  Then: .venv\Scripts\Activate" -ForegroundColor Yellow
    Write-Host "  Then: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Start Backend
Write-Host "  [1/2] Starting Backend (port 3000)..." -ForegroundColor Green
$backendCmd = "Set-Location '$ROOT'; . '$venvActivate'; python main.py"
$backend = Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -PassThru

Start-Sleep -Seconds 2

# Start Frontend
Write-Host "  [2/2] Starting Frontend (port 3001)..." -ForegroundColor Green
$frontendDir = Join-Path $ROOT "frontend"
$frontendCmd = "Set-Location '$frontendDir'; npm run dev"
$frontend = Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd -PassThru

Start-Sleep -Seconds 3

Write-Host ""
Write-Host "  Backend:  http://localhost:3000" -ForegroundColor Yellow
Write-Host "  Frontend: http://localhost:3001" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Press Enter to stop both servers..." -ForegroundColor DarkGray
Read-Host

# Cleanup
Write-Host "  Stopping servers..." -ForegroundColor Red
Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
Write-Host "  Stopped." -ForegroundColor Green
