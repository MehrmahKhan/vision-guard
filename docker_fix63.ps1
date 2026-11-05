# Redoes only the two 6.3 isolation screenshots, without Test-NetConnection,
# whose progress banner does not clear inside the VS Code terminal.
# Run this AFTER docker_auto.ps1 finishes, with the stack still up.

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }
$shotDir = Join-Path $root "evidence\docker"

function Grab($file) {
    Start-Sleep -Milliseconds 1500
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bitmap.Size)
    $bitmap.Save((Join-Path $shotDir $file), [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose(); $bitmap.Dispose()
    Write-Host ""
    Write-Host ("  [captured " + $file + "]") -ForegroundColor DarkGray
    Start-Sleep -Seconds 2
}

function PortOpen($hostname, $port, $timeoutMs) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($hostname, $port, $null, $null)
        $ok = $async.AsyncWaitHandle.WaitOne($timeoutMs, $false)
        if ($ok -and $client.Connected) { $client.EndConnect($async); return $true }
        return $false
    } catch { return $false } finally { $client.Close() }
}

Clear-Host
Write-Host ("=" * 78) -ForegroundColor Cyan
Write-Host "  6.3  The database port is not published to the host" -ForegroundColor Cyan
Write-Host ("=" * 78) -ForegroundColor Cyan
Write-Host ""
Write-Host "PS C:\mlops_a3> docker compose ps" -ForegroundColor White
Write-Host ""
docker compose ps --format "table {{.Service}}`t{{.Status}}`t{{.Ports}}"
Write-Host ""
Write-Host "Note the db row publishes nothing to the host, only 5432/tcp internally."
Write-Host ""
Write-Host "PS C:\mlops_a3> docker compose port db 5432" -ForegroundColor White
$portOut = (docker compose port db 5432 2>&1 | Out-String).Trim()
if ($portOut -eq "") { Write-Host "  (no output: nothing is published)" -ForegroundColor Yellow }
else { Write-Host ("  " + $portOut) -ForegroundColor Yellow }
Write-Host ""
Write-Host "Opening a TCP socket to localhost:5432 from the host:" -ForegroundColor White
$reachable = PortOpen "127.0.0.1" 5432 4000
Write-Host ("  connection succeeded : " + $reachable) -ForegroundColor Yellow
Write-Host ""
Write-Host "False is the correct and intended result. Postgres is unreachable" -ForegroundColor Green
Write-Host "from this machine because the compose file publishes no port for it." -ForegroundColor Green
Grab "13_6_3_closed.png"

Clear-Host
Write-Host ("=" * 78) -ForegroundColor Cyan
Write-Host "  6.3  But the web container reaches it by service name" -ForegroundColor Cyan
Write-Host ("=" * 78) -ForegroundColor Cyan
Write-Host ""
Write-Host "PS C:\mlops_a3> docker compose exec web python -c ..." -ForegroundColor White
Write-Host ""
docker compose exec -T web python -c "import socket; s = socket.create_connection(('db', 5432), 5); print('connected to db:5432 from inside the web container'); print('local  =', s.getsockname()); print('peer   =', s.getpeername()); s.close(); print('db resolves to', socket.gethostbyname('db'))"
Write-Host ""
Write-Host "The same port that was closed from the host is open from inside the" -ForegroundColor Green
Write-Host "compose network, addressed by the service name rather than an IP." -ForegroundColor Green
Grab "14_6_3_internal.png"

Clear-Host
Write-Host "Both 6.3 isolation screenshots re-captured cleanly." -ForegroundColor Green
Write-Host ("  " + (Join-Path $shotDir "13_6_3_closed.png"))
Write-Host ("  " + (Join-Path $shotDir "14_6_3_internal.png"))
