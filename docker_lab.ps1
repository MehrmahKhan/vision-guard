# Task 6 - the Docker lab, run in stages so you can screenshot each one.
#
# Usage, from C:\mlops_a3 :
#     .\docker_lab.ps1 warmup     # 6.1  Docker warm-up with nginx
#     .\docker_lab.ps1 build      # 6.2  build the web image, show the non-root user
#     .\docker_lab.ps1 compose    # 6.3  bring up the stack, prove the db port is closed
#     .\docker_lab.ps1 size       # 6.4  image size analysis
#     .\docker_lab.ps1 sizecmp    # 6.4  measured before/after vs a naive image
#     .\docker_lab.ps1 ops        # 6.5  start, stop, restart, logs, shells
#     .\docker_lab.ps1 break      # 6.5  break it on purpose, read the logs
#     .\docker_lab.ps1 fix        # 6.5  put it back
#
# Screenshot after each stage. Each one pauses at the end so you have time.

param([Parameter(Mandatory=$true)][string]$stage)

function Banner($text) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host $text -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ""
}

function Pause-Here($what) {
    Write-Host ""
    Write-Host ">>> SCREENSHOT NOW: $what" -ForegroundColor Yellow
    Read-Host ">>> press Enter when you have it"
}

if ($stage -eq "warmup") {
    Banner "6.1 - run a container and map a port"
    docker run -d --name warmup -p 8080:80 nginx:alpine
    docker ps
    Write-Host ""
    Write-Host "Open http://localhost:8080 in a browser, you should see the nginx page."
    Pause-Here "docker ps output AND the nginx page in the browser"

    Banner "6.1 - read its logs"
    docker logs warmup
    Pause-Here "the container logs"

    Banner "6.1 - look inside the container"
    docker exec warmup sh -c "whoami; ls /usr/share/nginx/html; cat /etc/os-release | head -2"
    Pause-Here "the output from inside the container"

    Banner "6.1 - mount a volume"
    docker rm -f warmup | Out-Null
    New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\nginx-test" | Out-Null
    "<h1>Served from a mounted volume on my laptop</h1>" | Out-File -Encoding ascii "$env:USERPROFILE\nginx-test\index.html"
    docker run -d --name warmup2 -p 8080:80 -v "$env:USERPROFILE\nginx-test:/usr/share/nginx/html:ro" nginx:alpine
    Start-Sleep -Seconds 2
    docker ps
    Write-Host ""
    Write-Host "Reload http://localhost:8080 , you should now see your own page."
    Pause-Here "the browser showing YOUR page, proving the volume mount"

    Banner "6.1 - list and remove"
    docker ps -a
    docker images
    docker rm -f warmup2 | Out-Null
    docker rmi nginx:alpine
    docker ps -a
    Pause-Here "the cleanup output"
    Write-Host "6.1 done." -ForegroundColor Green
}

elseif ($stage -eq "build") {
    Banner "6.2 - build the web image"
    docker build -t visionguard-web:latest .
    Pause-Here "the build output, note any CACHED layers"

    Banner "6.2 - prove the container does not run as root"
    docker run --rm visionguard-web:latest whoami
    docker run --rm visionguard-web:latest id
    Pause-Here "whoami showing appuser and id showing uid 10001"

    Banner "6.2 - rebuild to show the layer cache"
    Write-Host "Nothing changed, so every layer should say CACHED."
    docker build -t visionguard-web:latest .
    Pause-Here "the CACHED lines"
    Write-Host "6.2 done." -ForegroundColor Green
}

elseif ($stage -eq "compose") {
    Banner "6.3 - bring up the stack"
    Write-Host "Using the local model inside the container for this task, so the"
    Write-Host "mounted model volume is what is being demonstrated."
    $env:MODEL_API_URL = ""
    docker compose up -d --build
    Write-Host ""
    Write-Host "waiting for health checks ..."
    Start-Sleep -Seconds 45
    docker compose ps
    Pause-Here "docker compose ps showing both services healthy"

    Banner "6.3 - the database port is NOT published to the host"
    Write-Host "This command should FAIL, which is the point:"
    docker compose port db 5432
    Write-Host ""
    Write-Host "And from the host, nothing is listening on 5432:"
    Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet
    Pause-Here "the failed port lookup, proving the db is closed from outside"

    Banner "6.3 - but the web container CAN reach it by service name"
    docker compose exec web python -c "import socket; s=socket.create_connection(('db',5432),5); print('reached db:5432 from inside the web container'); s.close()"
    Pause-Here "the successful connection from inside the network"

    Banner "6.3 - the app is running on Postgres now"
    Write-Host "Open http://localhost:5000 , upload an image, then check History."
    Pause-Here "the app and its history page, now backed by Postgres"
    Write-Host "6.3 done." -ForegroundColor Green
}

elseif ($stage -eq "size") {
    Banner "6.4 - image size and the largest layers"
    docker images visionguard-web
    Write-Host ""
    docker history visionguard-web:latest --human --format "table {{.Size}}`t{{.CreatedBy}}" | Select-Object -First 12
    Pause-Here "the image size and the layer breakdown"
    Write-Host "6.4 done. For the before/after comparison, see the note at the end." -ForegroundColor Green
}

elseif ($stage -eq "ops") {
    Banner "6.5 - stop"
    docker compose stop
    docker compose ps
    Pause-Here "both services stopped"

    Banner "6.5 - start"
    docker compose start
    Start-Sleep -Seconds 20
    docker compose ps
    Pause-Here "both services running again"

    Banner "6.5 - restart just the web service"
    docker compose restart web
    Start-Sleep -Seconds 15
    docker compose ps
    Pause-Here "after the restart"

    Banner "6.5 - logs from each service"
    docker compose logs --tail 25 web
    Write-Host ""
    docker compose logs --tail 25 db
    Pause-Here "the logs from both services"

    Banner "6.5 - a shell inside each service"
    docker compose exec web sh -c "whoami; pwd; ls -la /app; ls -la /app/models"
    Write-Host ""
    docker compose exec db sh -c "whoami; psql -U visionguard -d visionguard -c '\dt'"
    Pause-Here "the output from inside both containers"
    Write-Host "6.5 operations done." -ForegroundColor Green
}

elseif ($stage -eq "break") {
    Banner "6.5 - breaking it on purpose: wrong database password"
    Write-Host "Starting the web service with a deliberately wrong password."
    docker compose stop web | Out-Null
    docker compose run --rm -e DATABASE_URL="postgresql://visionguard:WRONG_PASSWORD@db:5432/visionguard" web python -c "from app import db; db.init_db()"
    Write-Host ""
    Write-Host "Read the error above. Note that it is an AUTHENTICATION failure,"
    Write-Host "not a connection failure, which is what points at the credentials"
    Write-Host "rather than the network or the hostname."
    Pause-Here "the authentication error, this is your diagnosis evidence"
    Write-Host "Now run:  .\docker_lab.ps1 fix" -ForegroundColor Yellow
}

elseif ($stage -eq "fix") {
    Banner "6.5 - fixed: correct password, service healthy again"
    docker compose up -d
    Start-Sleep -Seconds 30
    docker compose ps
    docker compose logs --tail 15 web
    Pause-Here "the healthy service after the fix"
    Write-Host "6.5 done." -ForegroundColor Green
}

elseif ($stage -eq "sizecmp") {
    Banner "6.4 - measured before/after: base image choice"
    Write-Host "Pulling the full python:3.11 so the two bases can be compared."
    docker pull python:3.11
    docker pull python:3.11-slim
    docker images --filter "reference=python:3.11*" --format "table {{.Repository}}:{{.Tag}}`t{{.Size}}"
    Pause-Here "the two base images side by side, slim vs full"

    Banner "6.4 - build the naive image"
    Write-Host "This is Dockerfile.fat, the version with the three mistakes in it."
    docker build -f Dockerfile.fat -t visionguard-web:fat .
    Pause-Here "the naive build finishing"

    Banner "6.4 - the comparison"
    docker images visionguard-web --format "table {{.Tag}}`t{{.Size}}"
    Write-Host ""
    Write-Host "Largest layers in the OPTIMISED image:"
    docker history visionguard-web:latest --human --format "table {{.Size}}`t{{.CreatedBy}}" | Select-Object -First 8
    Write-Host ""
    Write-Host "Largest layers in the NAIVE image:"
    docker history visionguard-web:fat --human --format "table {{.Size}}`t{{.CreatedBy}}" | Select-Object -First 8
    Pause-Here "both sizes and both layer breakdowns, this is the 6.4 evidence"

    Banner "6.4 - cache behaviour, the other half of the saving"
    Write-Host "Touching app/app.py and rebuilding BOTH images."
    (Get-Item app\app.py).LastWriteTime = Get-Date
    Write-Host ""
    Write-Host "--- optimised: the pip install layer should stay CACHED ---"
    Measure-Command { docker build -t visionguard-web:latest . } | Select-Object TotalSeconds
    Write-Host ""
    Write-Host "--- naive: the pip install layer is invalidated and reruns ---"
    Measure-Command { docker build -f Dockerfile.fat -t visionguard-web:fat . } | Select-Object TotalSeconds
    Pause-Here "the two rebuild times, proving the layer ordering matters"

    Banner "6.4 - drop the naive image again"
    docker rmi visionguard-web:fat python:3.11
    docker images
    Pause-Here "the cleanup"
    Write-Host "6.4 done." -ForegroundColor Green
}

else {
    Write-Host "Unknown stage. Use one of: warmup build compose size sizecmp ops break fix" -ForegroundColor Red
}
