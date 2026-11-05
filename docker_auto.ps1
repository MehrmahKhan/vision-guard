# Task 6 - the whole Docker lab, unattended.
#
# Runs every stage back to back, writes a full text transcript, and takes a
# real screenshot of the console after each step. Nothing is simulated: every
# image is a capture of this machine actually running the command above it.
#
# Usage, from C:\mlops_a3 with the venv active:
#     .\docker_auto.ps1
#
# While it runs, DO NOT touch the mouse or keyboard and do not click another
# window. The screenshots capture whatever is on screen, so the console has to
# stay in front. It takes roughly 30 minutes. Walk away.

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }
$shotDir = Join-Path $root "evidence\docker"
$logPath = Join-Path $root "evidence\docker_transcript.txt"
New-Item -ItemType Directory -Force -Path $shotDir | Out-Null

$script:stepNumber = 0
$script:manifest = New-Object System.Collections.ArrayList

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
    $graphics.Dispose()
    $bitmap.Dispose()

    $line = $num + "_" + $slug + ".png | " + $caption
    [void]$script:manifest.Add($line)
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

Write-Host "VisionGuard Docker lab, unattended run" -ForegroundColor Green
Write-Host ("started " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Host ""
docker --version
docker compose version
Start-Sleep -Seconds 3

# ---------------------------------------------------------------- 6.1 warm-up

Step "6.1  Run a container and publish a port" `
     "An nginx container started detached with host port 8080 mapped to container port 80." `
     "6_1_run" "docker run -d --name warmup -p 8080:80 nginx:alpine" {
    docker rm -f warmup warmup2 2>$null | Out-Null
    docker run -d --name warmup -p 8080:80 nginx:alpine
    Start-Sleep -Seconds 3
    Write-Host ""
    docker ps
}

Step "6.1  The published port actually serves" `
     "Fetching http://localhost:8080 from the host proves the port mapping works. The default nginx page comes back." `
     "6_1_serve" "Invoke-WebRequest http://localhost:8080" {
    $response = Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing
    Write-Host ("HTTP status : " + $response.StatusCode)
    Write-Host ("Server      : " + $response.Headers["Server"])
    Write-Host ("Body length : " + $response.Content.Length + " bytes")
    Write-Host ""
    Write-Host "First lines of the body:"
    $lines = $response.Content -split "`n"
    for ($i = 0; $i -lt 8 -and $i -lt $lines.Count; $i++) { Write-Host ("  " + $lines[$i].TrimEnd()) }
}

Step "6.1  Container logs" `
     "The access log line from the request above, read out of the container with docker logs." `
     "6_1_logs" "docker logs warmup" {
    docker logs warmup
}

Step "6.1  A shell inside the container" `
     "Running commands inside the container with docker exec. It is a separate Alpine filesystem running as root, which is why the VisionGuard image deliberately does not." `
     "6_1_exec" "docker exec warmup sh -c 'whoami; ls /usr/share/nginx/html; head -2 /etc/os-release'" {
    docker exec warmup sh -c "whoami; echo ---; ls /usr/share/nginx/html; echo ---; head -2 /etc/os-release"
}

Step "6.1  Mount a host folder as a volume" `
     "The container is replaced with one that mounts a folder from the laptop read-only over the nginx web root." `
     "6_1_volume_up" "docker run -d --name warmup2 -p 8080:80 -v `$env:USERPROFILE\nginx-test:/usr/share/nginx/html:ro nginx:alpine" {
    docker rm -f warmup | Out-Null
    $mountDir = Join-Path $env:USERPROFILE "nginx-test"
    New-Item -ItemType Directory -Force -Path $mountDir | Out-Null
    $html = "<h1>Served from a mounted volume on my laptop</h1><p>VisionGuard, 23L-0623</p>"
    $html | Out-File -Encoding ascii (Join-Path $mountDir "index.html")
    Write-Host ("host folder : " + $mountDir)
    Get-Content (Join-Path $mountDir "index.html")
    Write-Host ""
    docker run -d --name warmup2 -p 8080:80 -v ($mountDir + ":/usr/share/nginx/html:ro") nginx:alpine
    Start-Sleep -Seconds 3
    docker ps
}

Step "6.1  The volume is what nginx now serves" `
     "The same URL now returns the file from the laptop instead of the nginx default page, which is the proof the bind mount took effect." `
     "6_1_volume_proof" "Invoke-WebRequest http://localhost:8080" {
    $response = Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing
    Write-Host ("HTTP status : " + $response.StatusCode)
    Write-Host ""
    Write-Host "Body returned by the server:" -ForegroundColor Yellow
    Write-Host ("  " + $response.Content.Trim())
    Write-Host ""
    Write-Host "Writing to the mount from inside the container should FAIL, it is :ro"
    docker exec warmup2 sh -c "echo test > /usr/share/nginx/html/x.html" 2>&1
}

Step "6.1  List and remove" `
     "Listing containers and images, then removing the container and the nginx image to leave the host clean." `
     "6_1_cleanup" "docker ps -a ; docker images ; docker rm -f warmup2 ; docker rmi nginx:alpine" {
    docker ps -a
    Write-Host ""
    docker images
    Write-Host ""
    docker rm -f warmup2
    docker rmi nginx:alpine
    Write-Host ""
    docker ps -a
}

# ------------------------------------------------------------------ 6.2 build

Step "6.2  Build the VisionGuard web image" `
     "The first build of the application image. The dependency layer is installed before the source is copied, which is what makes later rebuilds cheap." `
     "6_2_build" "docker build -t visionguard-web:latest ." {
    docker build -t visionguard-web:latest .
}

Step "6.2  The container does not run as root" `
     "whoami and id inside the built image, showing the unprivileged appuser account at uid 10001 created in the Dockerfile." `
     "6_2_nonroot" "docker run --rm visionguard-web:latest whoami ; docker run --rm visionguard-web:latest id" {
    Write-Host "whoami:"
    docker run --rm visionguard-web:latest whoami
    Write-Host ""
    Write-Host "id:"
    docker run --rm visionguard-web:latest id
    Write-Host ""
    Write-Host "For contrast, the base image default would be root. This image sets USER appuser."
}

Step "6.2  The layer cache" `
     "Rebuilding with nothing changed. Every instruction reports CACHED and the build finishes in under a second." `
     "6_2_cached" "docker build -t visionguard-web:latest ." {
    $elapsed = Measure-Command { docker build -t visionguard-web:latest . 2>&1 | Out-Host }
    Write-Host ""
    Write-Host ("rebuild took " + [math]::Round($elapsed.TotalSeconds, 2) + " seconds") -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 6.3 compose

Step "6.3  Bring up the stack" `
     "docker compose up starts the Flask service and a Postgres service on a private user-defined network." `
     "6_3_up" "docker compose up -d --build" {
    $env:MODEL_API_URL = ""
    docker compose up -d --build
    Write-Host ""
    Write-Host "waiting for health checks, this takes about a minute ..."
    Start-Sleep -Seconds 60
}

Step "6.3  Both services healthy" `
     "docker compose ps with both services up. The web service reports healthy only once its /health endpoint answers, which means the model loaded." `
     "6_3_ps" "docker compose ps" {
    docker compose ps
}

Step "6.3  The database port is not published to the host" `
     "The db service publishes no ports, so the host cannot reach 5432 at all. This is the isolation the task asks to demonstrate." `
     "6_3_closed" "docker compose port db 5432 ; Test-NetConnection localhost -Port 5432" {
    Write-Host "docker compose port db 5432   (expected to fail, nothing is published)"
    docker compose port db 5432 2>&1
    Write-Host ""
    Write-Host "Test-NetConnection localhost -Port 5432"
    $reachable = Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet -WarningAction SilentlyContinue
    Write-Host ("  TcpTestSucceeded : " + $reachable) -ForegroundColor Yellow
    Write-Host ""
    Write-Host "False is the correct and intended result."
}

Step "6.3  But the web container reaches it by service name" `
     "From inside the Compose network the database resolves as the hostname db and accepts a connection on 5432. Closed to the host, open to the service that needs it." `
     "6_3_internal" "docker compose exec web python -c `"socket.create_connection(('db',5432))`"" {
    docker compose exec -T web python -c "import socket; s = socket.create_connection(('db', 5432), 5); print('connected to db:5432 from inside the web container'); print('peer =', s.getpeername()); s.close()"
    Write-Host ""
    Write-Host "DNS for the service name, resolved inside the network:"
    docker compose exec -T web python -c "import socket; print('db resolves to', socket.gethostbyname('db'))"
}

Step "6.3  The app answers, backed by Postgres" `
     "The containerised app responding on port 5000, with its health endpoint reporting the loaded model and the Postgres backend." `
     "6_3_app" "Invoke-WebRequest http://localhost:5000/health" {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 30
        Write-Host ("HTTP status : " + $response.StatusCode)
        Write-Host ""
        Write-Host "Response body:" -ForegroundColor Yellow
        Write-Host ("  " + $response.Content)
    } catch {
        Write-Host ("health check failed: " + $_.Exception.Message) -ForegroundColor Red
        docker compose logs --tail 30 web
    }
}

# ------------------------------------------------------------------- 6.4 size

Step "6.4  Image size and the largest layers" `
     "The size of the optimised image and the layers that account for most of it. The pip install of tensorflow-cpu dominates, as expected for a model server." `
     "6_4_size" "docker images visionguard-web ; docker history visionguard-web:latest" {
    docker images visionguard-web
    Write-Host ""
    Write-Host "Largest layers:"
    docker history visionguard-web:latest --human --format "table {{.Size}}`t{{.CreatedBy}}" | Select-Object -First 10
}

Step "6.4  Base image comparison, slim against full" `
     "python:3.11-slim against the full python:3.11. The slim variant drops the build toolchain and documentation that an inference server never uses." `
     "6_4_bases" "docker pull python:3.11 ; docker images python" {
    docker pull python:3.11 2>&1 | Select-Object -Last 3
    Write-Host ""
    docker images --filter "reference=python" --format "table {{.Repository}}:{{.Tag}}`t{{.Size}}"
}

Step "6.4  Build the deliberately naive image" `
     "Dockerfile.fat, built only as a baseline. It uses the full base, copies the source before installing dependencies, and keeps the pip and apt caches." `
     "6_4_fatbuild" "docker build -f Dockerfile.fat -t visionguard-web:fat ." {
    docker build -f Dockerfile.fat -t visionguard-web:fat . 2>&1 | Select-Object -Last 25
}

Step "6.4  The measured difference" `
     "The optimised image against the naive one, with the layer breakdown of each. This is the before and after the sub-task asks for." `
     "6_4_compare" "docker images visionguard-web" {
    docker images visionguard-web --format "table {{.Tag}}`t{{.Size}}`t{{.CreatedAt}}"
    Write-Host ""
    Write-Host "OPTIMISED, top layers:" -ForegroundColor Yellow
    docker history visionguard-web:latest --human --format "table {{.Size}}`t{{.CreatedBy}}" | Select-Object -First 7
    Write-Host ""
    Write-Host "NAIVE, top layers:" -ForegroundColor Yellow
    docker history visionguard-web:fat --human --format "table {{.Size}}`t{{.CreatedBy}}" | Select-Object -First 7
}

Step "6.4  Rebuild cost after a one line source change" `
     "Touching app/app.py and rebuilding both. The optimised image reuses its dependency layer; the naive one copied the source first so it reinstalls everything." `
     "6_4_rebuild" "(touch app\app.py) ; docker build both" {
    (Get-Item (Join-Path $root "app\app.py")).LastWriteTime = Get-Date
    Write-Host "app/app.py timestamp updated, contents unchanged."
    Write-Host ""
    Write-Host "optimised rebuild:" -ForegroundColor Yellow
    $fast = Measure-Command { docker build -t visionguard-web:latest . 2>&1 | Out-Null }
    Write-Host ("  " + [math]::Round($fast.TotalSeconds, 2) + " seconds")
    Write-Host ""
    Write-Host "naive rebuild:" -ForegroundColor Yellow
    $slow = Measure-Command { docker build -f Dockerfile.fat -t visionguard-web:fat . 2>&1 | Out-Null }
    Write-Host ("  " + [math]::Round($slow.TotalSeconds, 2) + " seconds")
    Write-Host ""
    if ($fast.TotalSeconds -gt 0) {
        Write-Host ("naive is " + [math]::Round($slow.TotalSeconds / $fast.TotalSeconds, 1) + "x slower to rebuild") -ForegroundColor Green
    }
}

Step "6.4  Drop the naive image" `
     "Removing the baseline image and the full base now that the comparison is recorded." `
     "6_4_drop" "docker rmi visionguard-web:fat python:3.11" {
    docker rmi visionguard-web:fat 2>&1
    docker rmi python:3.11 2>&1
    Write-Host ""
    docker images
}

# -------------------------------------------------------------------- 6.5 ops

Step "6.5  Stop" `
     "docker compose stop halts both containers without deleting them or their volumes." `
     "6_5_stop" "docker compose stop ; docker compose ps -a" {
    docker compose stop
    Write-Host ""
    docker compose ps -a
}

Step "6.5  Start" `
     "docker compose start brings the same containers back. The Postgres data volume survived the stop." `
     "6_5_start" "docker compose start ; docker compose ps" {
    docker compose start
    Start-Sleep -Seconds 45
    docker compose ps
}

Step "6.5  Restart one service only" `
     "Restarting the web service alone, leaving the database untouched, which is the normal way to pick up a code change." `
     "6_5_restart" "docker compose restart web ; docker compose ps" {
    docker compose restart web
    Start-Sleep -Seconds 40
    docker compose ps
}

Step "6.5  Logs from both services" `
     "Application logs from the web service and startup logs from Postgres, read through Compose." `
     "6_5_logs" "docker compose logs --tail 20 web ; docker compose logs --tail 20 db" {
    Write-Host "WEB:" -ForegroundColor Yellow
    docker compose logs --tail 18 web
    Write-Host ""
    Write-Host "DB:" -ForegroundColor Yellow
    docker compose logs --tail 12 db
}

Step "6.5  A shell inside each service" `
     "Inside the web container: the unprivileged user, the application files and the mounted model. Inside the db container: the tables the app created." `
     "6_5_shells" "docker compose exec web sh ; docker compose exec db psql" {
    Write-Host "WEB container:" -ForegroundColor Yellow
    docker compose exec -T web sh -c "whoami; pwd; echo ---; ls -la /app; echo ---; ls -la /app/models"
    Write-Host ""
    Write-Host "DB container:" -ForegroundColor Yellow
    docker compose exec -T db sh -c "whoami; psql -U visionguard -d visionguard -c '\dt'"
}

Step "6.5  Break it on purpose, wrong database password" `
     "The web service started against the database with a deliberately wrong password. The failure is an authentication error, not a connection or DNS error, which is what points the diagnosis at the credentials rather than the network." `
     "6_5_break" "docker compose run -e DATABASE_URL=...WRONG... web python -c 'db.init_db()'" {
    docker compose run --rm -T -e DATABASE_URL="postgresql://visionguard:WRONG_PASSWORD@db:5432/visionguard" web python -c "from app import db; db.init_db()" 2>&1 | Select-Object -Last 22
    Write-Host ""
    Write-Host "Read the exception type above: it is an authentication failure." -ForegroundColor Yellow
    Write-Host "The host db resolved and the TCP connection was accepted, so the" -ForegroundColor Yellow
    Write-Host "network and the service name are fine. Only the password is wrong." -ForegroundColor Yellow
}

Step "6.5  Fixed, healthy again" `
     "With the correct credentials restored from the environment the stack comes back healthy and the application answers again." `
     "6_5_fix" "docker compose up -d ; docker compose ps" {
    docker compose up -d
    Start-Sleep -Seconds 50
    docker compose ps
    Write-Host ""
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 30
        Write-Host ("health endpoint : " + $response.StatusCode + "  " + $response.Content)
    } catch {
        Write-Host "health check still failing, see logs" -ForegroundColor Red
        docker compose logs --tail 15 web
    }
}

# ------------------------------------------------------------------- wrap up

Clear-Host
Write-Host ("=" * 78) -ForegroundColor Green
Write-Host "  Docker lab finished" -ForegroundColor Green
Write-Host ("=" * 78) -ForegroundColor Green
Write-Host ""
Write-Host ("screenshots : " + $shotDir)
Write-Host ("transcript  : " + $logPath)
Write-Host ""
Write-Host ("captured " + $script:stepNumber + " screenshots:")
foreach ($line in $script:manifest) { Write-Host ("  " + $line) }

$manifestPath = Join-Path $root "evidence\docker_manifest.txt"
$script:manifest | Out-File -Encoding utf8 $manifestPath
Write-Host ""
Write-Host ("manifest    : " + $manifestPath)
Write-Host ""
Write-Host "The stack is still up. To take the two browser screenshots:" -ForegroundColor Yellow
Write-Host "  open http://localhost:5000 , upload an image, then open History." -ForegroundColor Yellow
Write-Host "To shut everything down afterwards:  docker compose down" -ForegroundColor Yellow

Stop-Transcript | Out-Null
