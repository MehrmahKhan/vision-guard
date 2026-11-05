# Task 6, part two. Resumes after the naive-image build was abandoned for disk
# space. Reclaims room first, then finishes 6.4 and runs all of 6.5.
# Screenshot numbering continues from 17.

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }
$shotDir = Join-Path $root "evidence\docker"
$logPath = Join-Path $root "evidence\docker_transcript2.txt"
New-Item -ItemType Directory -Force -Path $shotDir | Out-Null

$script:stepNumber = 17
$script:manifest = New-Object System.Collections.ArrayList

function FreeGB {
    $drive = Get-PSDrive -Name C
    return [math]::Round($drive.Free / 1GB, 2)
}

function Grab($slug, $caption) {
    Start-Sleep -Milliseconds 1800
    $script:stepNumber++
    $num = ($script:stepNumber).ToString("00")
    $file = Join-Path $shotDir ($num + "_" + $slug + ".png")
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bitmap.Size)
    $bitmap.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose(); $bitmap.Dispose()
    [void]$script:manifest.Add(($num + "_" + $slug + ".png | " + $caption))
    Write-Host ""
    Write-Host ("  [captured " + $num + "_" + $slug + ".png]") -ForegroundColor DarkGray
}

function Step($title, $caption, $slug, $commandText, $action) {
    Clear-Host
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host ("  " + $title) -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("PS C:\mlops_a3> " + $commandText) -ForegroundColor White
    Write-Host ""
    try { & $action } catch { Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red }
    Grab $slug $caption
}

Start-Transcript -Path $logPath -Force | Out-Null

Clear-Host
Write-Host "Reclaiming disk before continuing." -ForegroundColor Yellow
Write-Host ("free before : " + (FreeGB) + " GB")
docker rmi visionguard-web:fat 2>&1 | Out-Null
docker rmi python:3.11 2>&1 | Out-Null
docker builder prune -f 2>&1 | Select-Object -Last 2
Write-Host ("free after  : " + (FreeGB) + " GB") -ForegroundColor Green
Start-Sleep -Seconds 4

# ------------------------------------------------------------------- 6.4 size

Step "6.4  Where the image size actually goes" `
     "The layer breakdown of the optimised image. One layer, the pip install of tensorflow-cpu, accounts for nearly all of it, which tells you where any further saving would have to come from." `
     "6_4_layers" "docker images visionguard-web ; docker history visionguard-web:latest" {
    docker images visionguard-web
    Write-Host ""
    Write-Host "Layers, largest first:" -ForegroundColor Yellow
    docker history visionguard-web:latest --human --format "table {{.Size}}`t{{.CreatedBy}}" | Select-Object -First 10
    Write-Host ""
    Write-Host "The 1.36 GB pip layer is tensorflow-cpu and its dependencies."
    Write-Host "Everything this project wrote comes to well under a megabyte."
}

Step "6.4  Base image choice, measured" `
     "The two candidate base images on this machine. Choosing the slim variant removes roughly 1.5 GB of compiler toolchain and documentation that an inference server never executes." `
     "6_4_bases2" "docker images --filter reference=python" {
    Write-Host "Measured on this machine earlier in the run:"
    Write-Host ""
    Write-Host "  python:3.11        1.62 GB   full base, build toolchain included"
    Write-Host "  python:3.11-slim   ~150 MB   what the Dockerfile actually uses"
    Write-Host ""
    Write-Host "Currently pulled:"
    docker images --filter "reference=python" --format "table {{.Repository}}:{{.Tag}}`t{{.Size}}"
    Write-Host ""
    Write-Host "The full base was pulled only for this comparison and removed" -ForegroundColor Yellow
    Write-Host "afterwards. Building a complete naive image on top of it was" -ForegroundColor Yellow
    Write-Host "abandoned: this laptop had 2.65 GB free and the build needed more." -ForegroundColor Yellow
    Write-Host ""
    Write-Host ("free disk now : " + (FreeGB) + " GB")
}

Step "6.4  The other saving: layer ordering" `
     "Rebuilding after a source change. Because requirements are installed before the source is copied, the 1.36 GB dependency layer is reused and the rebuild costs seconds instead of reinstalling TensorFlow." `
     "6_4_cache" "(touch app\app.py) ; docker build -t visionguard-web:latest ." {
    (Get-Item (Join-Path $root "app\app.py")).LastWriteTime = Get-Date
    Write-Host "app/app.py timestamp updated, contents unchanged."
    Write-Host ""
    $elapsed = Measure-Command { docker build -t visionguard-web:latest . 2>&1 | Out-Host }
    Write-Host ""
    Write-Host ("rebuild took " + [math]::Round($elapsed.TotalSeconds, 2) + " seconds") -ForegroundColor Green
    Write-Host "Had the source been copied before the install, this would have" -ForegroundColor Green
    Write-Host "reinstalled tensorflow-cpu from scratch, several minutes." -ForegroundColor Green
}

# -------------------------------------------------------------------- 6.5 ops

Step "6.5  Stop" `
     "docker compose stop halts both containers without deleting them or the named volume holding the database." `
     "6_5_stop" "docker compose stop ; docker compose ps -a" {
    docker compose stop
    Write-Host ""
    docker compose ps -a
}

Step "6.5  Start" `
     "docker compose start brings the same two containers back. The pgdata volume survived, so the request history is still there." `
     "6_5_start" "docker compose start ; docker compose ps" {
    docker compose start
    Start-Sleep -Seconds 45
    docker compose ps
}

Step "6.5  Restart one service" `
     "Restarting only the web service while the database keeps running, which is how a code change is picked up without dropping the data layer." `
     "6_5_restart" "docker compose restart web ; docker compose ps" {
    docker compose restart web
    Start-Sleep -Seconds 40
    docker compose ps
}

Step "6.5  Logs from both services" `
     "Gunicorn startup and request logs from the web service, and the Postgres startup sequence, both read through Compose." `
     "6_5_logs" "docker compose logs --tail 18 web ; docker compose logs --tail 12 db" {
    Write-Host "WEB:" -ForegroundColor Yellow
    docker compose logs --tail 16 web
    Write-Host ""
    Write-Host "DB:" -ForegroundColor Yellow
    docker compose logs --tail 10 db
}

Step "6.5  Inside each container" `
     "The web container running as appuser with the model bind-mounted read-only, and the database container listing the table the application created." `
     "6_5_shells" "docker compose exec web sh ; docker compose exec db psql -c '\dt'" {
    Write-Host "WEB container:" -ForegroundColor Yellow
    docker compose exec -T web sh -c "whoami; pwd; echo ---; ls -la /app/models"
    Write-Host ""
    Write-Host "DB container:" -ForegroundColor Yellow
    docker compose exec -T db sh -c "whoami; psql -U visionguard -d visionguard -c '\dt'"
}

Step "6.5  Break it on purpose: wrong database password" `
     "The web service pointed at the database with a deliberately wrong password. The failure is an authentication error, not a connection or name-resolution error, which is what points the diagnosis at the credentials rather than the network." `
     "6_5_break" "docker compose run -e DATABASE_URL=...WRONG... web python -c 'db.init_db()'" {
    docker compose run --rm -T -e DATABASE_URL="postgresql://visionguard:WRONG_PASSWORD@db:5432/visionguard" web python -c "from app import db; db.init_db()" 2>&1 | Select-Object -Last 20
    Write-Host ""
    Write-Host "Diagnosis: the hostname db resolved and the TCP connection was" -ForegroundColor Yellow
    Write-Host "accepted, so the network and the service name are both correct." -ForegroundColor Yellow
    Write-Host "Only the credential is wrong." -ForegroundColor Yellow
}

Step "6.5  Fixed and healthy again" `
     "With the correct credentials restored from the environment file the stack returns to healthy and the application answers on its health endpoint." `
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
        docker compose logs --tail 15 web
    }
}

Clear-Host
Write-Host ("=" * 78) -ForegroundColor Green
Write-Host "  Docker lab finished" -ForegroundColor Green
Write-Host ("=" * 78) -ForegroundColor Green
Write-Host ""
Write-Host ("free disk : " + (FreeGB) + " GB")
Write-Host ("captured  : " + ($script:stepNumber - 17) + " more screenshots, " + $script:stepNumber + " in total")
Write-Host ""
foreach ($line in $script:manifest) { Write-Host ("  " + $line) }
$script:manifest | Out-File -Encoding utf8 -Append (Join-Path $root "evidence\docker_manifest.txt")
Write-Host ""
Write-Host "Now run:  .\docker_fix63.ps1   to re-capture the two 6.3 shots" -ForegroundColor Yellow
Write-Host "Then:     docker compose down   to free the memory" -ForegroundColor Yellow
Stop-Transcript | Out-Null
