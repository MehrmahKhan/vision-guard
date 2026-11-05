# Task 7 - the five API test cases against the deployed service.
# Run from C:\mlops_a3 :   .\api_tests.ps1
# Screenshot the output. Each test prints the request and the full response.

$BASE = "https://vision-guard-kappa.vercel.app"
$KEY  = "e37Xfl_QPd6F0eOU2GKazaHUIGzmy2OYRCex2RJnRlI"

function Banner($n, $text) {
    Write-Host ""
    Write-Host ("-" * 72) -ForegroundColor Cyan
    Write-Host "TEST $n : $text" -ForegroundColor Cyan
    Write-Host ("-" * 72) -ForegroundColor Cyan
}

# Copy one real and one AI image to simple names in the current folder, so
# curl never has to deal with long or awkward paths.
$realSource = Get-ChildItem "data\raw\RealArt" -File -Include *.jpg,*.jpeg,*.png -Recurse | Select-Object -First 1
$aiSource   = Get-ChildItem "data\raw\AiArtData" -File -Include *.jpg,*.jpeg,*.png -Recurse | Select-Object -First 1

if ($null -eq $realSource) { Write-Host "Could not find a real image" -ForegroundColor Red; exit 1 }
if ($null -eq $aiSource)   { Write-Host "Could not find an AI image" -ForegroundColor Red; exit 1 }

Copy-Item $realSource.FullName -Destination "test_real.jpg" -Force
Copy-Item $aiSource.FullName   -Destination "test_ai.jpg"   -Force
"this is not an image" | Out-File -Encoding ascii "test_broken.jpg"

Write-Host "Target     : $BASE"
Write-Host "Real image : $($realSource.Name)  -> test_real.jpg"
Write-Host "AI image   : $($aiSource.Name)  -> test_ai.jpg"

Banner 1 "GET /health  (no API key needed, expect 200)"
curl.exe -sS -i "$BASE/health"
Write-Host ""

Banner 2 "POST /predict with a REAL photograph (expect 200, verdict REAL)"
curl.exe -sS -i -X POST -H "X-API-Key: $KEY" -F "file=@test_real.jpg" "$BASE/predict"
Write-Host ""

Banner 3 "POST /predict with an AI-GENERATED image (expect 200, verdict AI)"
curl.exe -sS -i -X POST -H "X-API-Key: $KEY" -F "file=@test_ai.jpg" "$BASE/predict"
Write-Host ""

Banner 4 "POST /predict with NO API key (expect 401 Unauthorized)"
curl.exe -sS -i -X POST -F "file=@test_real.jpg" "$BASE/predict"
Write-Host ""

Banner 5 "POST /predict with a non-image renamed .jpg (expect 422)"
curl.exe -sS -i -X POST -H "X-API-Key: $KEY" -F "file=@test_broken.jpg" "$BASE/predict"
Write-Host ""

Write-Host ("=" * 72) -ForegroundColor Green
Write-Host "All five tests complete. Screenshot the output above." -ForegroundColor Green
Write-Host ("=" * 72) -ForegroundColor Green
