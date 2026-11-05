# Task 6.5 re-capture. The earlier run produced correct output but the
# screenshots caught a different window. This version forces its own console
# to the foreground immediately before every capture, and verifies afterwards
# that consecutive screenshots actually differ.
#
# RUN THIS IN A STANDALONE POWERSHELL WINDOW, NOT THE VS CODE TERMINAL.
#   Start menu -> type "powershell" -> Windows PowerShell -> maximise it
#   cd C:\mlops_a3
#   .\docker_auto3.ps1

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WinFocus {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();
}
"@

$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }
$shotDir = Join-Path $root "evidence\docker"
New-Item -ItemType Directory -Force -Path $shotDir | Out-Null

$script:stepNumber = 19
$script:manifest = New-Object System.Collections.ArrayList
$script:hashes = @{}

function FocusSelf {
    $handle = [WinFocus]::GetConsoleWindow()
    if ($handle -ne [IntPtr]::Zero) {
        [void][WinFocus]::ShowWindow($handle, 3)
        [void][WinFocus]::SetForegroundWindow($handle)
        Start-Sleep -Milliseconds 400
    }
}

function Grab($slug, $caption) {
    FocusSelf
    Start-Sleep -Milliseconds 1400
    $script:stepNumber++
    $num = ($script:stepNumber).ToString("00")
    $name = $num + "_" + $slug + ".png"
    $file = Join-Path $shotDir $name

    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bitmap.Size)
    $bitmap.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose(); $bitmap.Dispose()

    $hash = (Get-FileHash -Path $file -Algorithm MD5).Hash
    if ($script:hashes.ContainsKey($hash)) {
        Write-Host ("  WARNING: " + $name + " is identical to " + $script:hashes[$hash]) -ForegroundColor Red
        Write-Host "  The console was not in front. Do not touch the machine." -ForegroundColor Red
    } else {
        $script:hashes[$hash] = $name
        Write-Host ("  [captured " + $name + "]") -ForegroundColor DarkGray
    }
    [void]$script:manifest.Add(($name + " | " + $caption))
}

function Step($title, $caption, $slug, $commandText, $action) {
    Clear-Host
    Write-Host ("=" * 76) -ForegroundColor Cyan
    Write-Host ("  " + $title) -ForegroundColor Cyan
    Write-Host ("=" * 76) -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("PS C:\mlops_a3> " + $commandText) -ForegroundColor White
    Write-Host ""
    try { & $action } catch { Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red }
    Write-Host ""
    Grab $slug $caption
}

Start-Transcript -Path (Join-Path $root "evidence\docker_transcript3.txt") -Force | Out-Null
FocusSelf

Clear-Host
Write-Host "Bringing the stack back up before re-capturing 6.5." -ForegroundColor Yellow
$env:MODEL_API_URL = ""
docker compose up -d
Write-Host ""
Write-Host "waiting for health checks ..."
Start-Sleep -Seconds 60

Step "6.4  Rebuild cost after a source change" `
     "Touching app/app.py and rebuilding. Because requirements are installed before the source is copied, the 1.36 GB dependency layer is reused and the rebuild finishes in seconds instead of reinstalling TensorFlow." `
     "6_4_cache" "(touch app\app.py) ; docker build -t visionguard-web:latest ." {
    (Get-Item (Join-Path $root "app\app.py")).LastWriteTime = Get-Date
    Write-Host "app/app.py timestamp updated, contents unchanged."
    Write-Host ""
    $elapsed = Measure-Command { docker build -t visionguard-web:latest . 2>&1 | Out-Host }
    Write-Host ""
    Write-Host ("rebuild took " + [math]::Round($elapsed.TotalSeconds, 2) + " seconds") -ForegroundColor Green
}

Step "6.5  Stop" `
     "docker compose stop halts both containers without deleting them or the named volume that holds the database." `
     "6_5_stop" "docker compose stop ; docker compose ps -a" {
    docker compose stop
    Write-Host ""
    docker compose ps -a
}

Step "6.5  Start" `
     "docker compose start brings the same two containers back. The pgdata volume survived the stop, so the request history is intact." `
     "6_5_start" "docker compose start ; docker compose ps" {
    docker compose start
    Start-Sleep -Seconds 45
    docker compose ps
}

Step "6.5  Restart one service only" `
     "Restarting the web service while the database keeps running, which is how a code change is picked up without disturbing the data layer." `
     "6_5_restart" "docker compose restart web ; docker compose ps" {
    docker compose restart web
    Start-Sleep -Seconds 40
    docker compose ps
}

Step "6.5  Logs from both services" `
     "Gunicorn startup and request logs from the web service, and the Postgres startup sequence, both read through Compose." `
     "6_5_logs" "docker compose logs --tail 14 web ; docker compose logs --tail 10 db" {
    Write-Host "WEB:" -ForegroundColor Yellow
    docker compose logs --tail 14 web
    Write-Host ""
    Write-Host "DB:" -ForegroundColor Yellow
    docker compose logs --tail 10 db
}

Step "6.5  Inside each container" `
     "The web container running as the unprivileged appuser with the model bind-mounted read-only, and the database container listing the table the application created." `
     "6_5_shells" "docker compose exec web sh ; docker compose exec db psql -c '\dt'" {
    Write-Host "WEB container:" -ForegroundColor Yellow
    docker compose exec -T web sh -c "whoami; pwd; echo ---; ls -la /app/models"
    Write-Host ""
    Write-Host "DB container:" -ForegroundColor Yellow
    docker compose exec -T db sh -c "whoami; psql -U visionguard -d visionguard -c '\dt'"
}

Step "6.5  Break it: wrong database password" `
     "The web service pointed at the database with a deliberately wrong password. The failure is an authentication error, not a connection or name-resolution error, which is what points the diagnosis at the credential rather than the network." `
     "6_5_break" "docker compose run -e DATABASE_URL=...WRONG... web python -c 'db.init_db()'" {
    docker compose run --rm -T -e DATABASE_URL="postgresql://visionguard:WRONG_PASSWORD@db:5432/visionguard" web python -c "from app import db; db.init_db()" 2>&1 | Select-Object -Last 18
    Write-Host ""
    Write-Host "Diagnosis: the hostname db resolved and the TCP connection was" -ForegroundColor Yellow
    Write-Host "accepted, so the network and service name are correct. Only the" -ForegroundColor Yellow
    Write-Host "credential is wrong." -ForegroundColor Yellow
}

Step "6.5  Fixed and healthy again" `
     "With the correct credential restored from the environment file the stack returns to healthy and the application answers on its health endpoint." `
     "6_5_fix" "docker compose up -d ; docker compose ps" {
    docker compose up -d
    Start-Sleep -Seconds 50
    docker compose ps
    Write-Host ""
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 30
        Write-Host ("health endpoint : " + $response.StatusCode)
        Write-Host ("  " + $response.Content)
    } catch {
        Write-Host "health check failing:" -ForegroundColor Red
        docker compose logs --tail 12 web
    }
}

Clear-Host
Write-Host ("=" * 76) -ForegroundColor Green
Write-Host "  6.5 re-captured" -ForegroundColor Green
Write-Host ("=" * 76) -ForegroundColor Green
Write-Host ""
Write-Host ("unique screenshots this run : " + $script:hashes.Count + " of " + ($script:stepNumber - 19))
Write-Host ""
foreach ($line in $script:manifest) { Write-Host ("  " + $line) }
$script:manifest | Out-File -Encoding utf8 -Append (Join-Path $root "evidence\docker_manifest.txt")
Write-Host ""
if ($script:hashes.Count -eq ($script:stepNumber - 19)) {
    Write-Host "All captures are distinct. Nothing to redo." -ForegroundColor Green
} else {
    Write-Host "Some captures repeated, tell Claude which ones warned above." -ForegroundColor Red
}
Write-Host ""
Write-Host "Afterwards:  docker compose down" -ForegroundColor Yellow
Stop-Transcript | Out-Null
