# Sub-task 1.4, the toy half. Creates a throwaway folder, tracks a small CSV
# with DVC, changes it, commits both versions, goes back to the old one, then
# pushes to a remote, deletes the local data and pulls it back.
#
# RUN IN A STANDALONE POWERSHELL WINDOW, MAXIMISED. Do not touch the machine.
#   cd C:\mlops_a3
#   .\dvc_lab.ps1

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System; using System.Runtime.InteropServices;
public class WF {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();
}
"@

$root = $PSScriptRoot; if (-not $root) { $root = (Get-Location).Path }
$shotDir = Join-Path $root "evidence\dvc"
New-Item -ItemType Directory -Force -Path $shotDir | Out-Null
$lab = Join-Path $env:USERPROFILE "dvc_toy_lab"
$remote = Join-Path $env:USERPROFILE "dvc_toy_remote"

$script:n = 0
$script:seen = @{}
$script:man = New-Object System.Collections.ArrayList

function Focus() {
  $h = [WF]::GetConsoleWindow()
  if ($h -ne [IntPtr]::Zero) { [void][WF]::ShowWindow($h,3); [void][WF]::SetForegroundWindow($h); Start-Sleep -Milliseconds 350 }
}
function Grab($slug, $cap) {
  Focus; Start-Sleep -Milliseconds 1300
  $script:n++
  $name = ($script:n).ToString("00") + "_" + $slug + ".png"
  $file = Join-Path $shotDir $name
  $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
  $bm = New-Object System.Drawing.Bitmap $b.Width, $b.Height
  $g = [System.Drawing.Graphics]::FromImage($bm)
  $g.CopyFromScreen($b.X, $b.Y, 0, 0, $bm.Size)
  $bm.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose(); $bm.Dispose()
  $hash = (Get-FileHash $file -Algorithm MD5).Hash
  if ($script:seen.ContainsKey($hash)) { Write-Host ("  WARNING: " + $name + " duplicates " + $script:seen[$hash]) -ForegroundColor Red }
  else { $script:seen[$hash] = $name; Write-Host ("  [captured " + $name + "]") -ForegroundColor DarkGray }
  [void]$script:man.Add($name + " | " + $cap)
}
function Step($title, $cap, $slug, $cmd, $act) {
  Clear-Host
  Write-Host ("=" * 76) -ForegroundColor Cyan
  Write-Host ("  " + $title) -ForegroundColor Cyan
  Write-Host ("=" * 76) -ForegroundColor Cyan
  Write-Host ""
  Write-Host ("PS> " + $cmd) -ForegroundColor White
  Write-Host ""
  try { & $act } catch { Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red }
  Write-Host ""
  Grab $slug $cap
}

Start-Transcript -Path (Join-Path $root "evidence\dvc_transcript.txt") -Force | Out-Null
Remove-Item -Recurse -Force $lab, $remote -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $lab, $remote | Out-Null
Set-Location $lab

Step "1.4 toy  Initialise Git and DVC in an empty folder" `
     "A throwaway folder with Git and DVC initialised. DVC creates a .dvc directory for its own config and cache, alongside the Git repository." `
     "init" "git init ; dvc init ; dir" {
  git init 2>&1 | Out-Host
  git config user.name "Mehrmah Khan"
  git config user.email "mehrmahkhan1@gmail.com"
  dvc init 2>&1 | Select-Object -Last 8
  Write-Host ""
  Get-ChildItem -Force | Select-Object Mode, Name | Format-Table -AutoSize
}

Step "1.4 toy  Create a small CSV and track it with DVC" `
     "Version 1 of the data, three rows, tracked with dvc add. DVC moves the file into its cache and leaves a small pointer file behind." `
     "add" "dvc add readings.csv" {
  "sensor,value`na,1`nb,2`nc,3" | Out-File -Encoding ascii readings.csv
  Write-Host "readings.csv, version 1:"
  Get-Content readings.csv
  Write-Host ""
  dvc add readings.csv 2>&1 | Select-Object -Last 8
  Write-Host ""
  Get-ChildItem -Name
}

Step "1.4 toy  What the pointer file actually holds" `
     "The .dvc pointer file. It records only an md5 of the content, the size and the path. That is why it is safe to commit to Git while the data itself is not: the pointer is a few hundred bytes and never changes size, however large the data grows." `
     "pointer" "type readings.csv.dvc ; git add -A ; git commit" {
  Get-Content readings.csv.dvc
  Write-Host ""
  Write-Host ("pointer file size : " + (Get-Item readings.csv.dvc).Length + " bytes")
  Write-Host ("data file size    : " + (Get-Item readings.csv).Length + " bytes")
  Write-Host ""
  Write-Host ".gitignore written by DVC so Git never sees the data:"
  Get-Content .gitignore
  Write-Host ""
  git add -A 2>&1 | Out-Null
  git commit -q -m "data v1" 2>&1 | Out-Host
  git log --oneline
}

Step "1.4 toy  Change the data and commit version 2" `
     "Version 2, with two extra rows. dvc add rewrites the pointer with a new hash, and Git records that change. The data never enters Git, only the hash does." `
     "v2" "dvc add readings.csv ; git commit -m 'data v2'" {
  "sensor,value`na,1`nb,2`nc,3`nd,4`ne,5" | Out-File -Encoding ascii readings.csv
  Write-Host "readings.csv, version 2:"
  Get-Content readings.csv
  Write-Host ""
  dvc add readings.csv 2>&1 | Select-Object -Last 4
  Write-Host ""
  Write-Host "the pointer now holds a different hash:"
  Get-Content readings.csv.dvc
  git add -A 2>&1 | Out-Null
  git commit -q -m "data v2" 2>&1 | Out-Host
  git log --oneline
}

Step "1.4 toy  Go back to version 1" `
     "Checking out the earlier commit restores the old pointer, and dvc checkout then restores the matching data from the cache. Five rows become three again. Git moved the pointer, DVC moved the data." `
     "rollback" "git checkout HEAD~1 -- readings.csv.dvc ; dvc checkout" {
  git checkout HEAD~1 -- readings.csv.dvc 2>&1 | Out-Host
  dvc checkout 2>&1 | Select-Object -Last 4
  Write-Host ""
  Write-Host "readings.csv is now:"
  Get-Content readings.csv
  Write-Host ""
  Write-Host "Back to version 1 without ever storing the data in Git." -ForegroundColor Green
  git checkout HEAD -- readings.csv.dvc 2>&1 | Out-Null
  dvc checkout 2>&1 | Out-Null
}

Step "1.4 toy  Add a remote, push, delete locally, pull it back" `
     "The full round trip. The data is pushed to a remote, then deleted from both the working folder and the local cache, so nothing is left. dvc pull fetches it back from the remote by hash." `
     "roundtrip" "dvc remote add -d store ... ; dvc push ; rm data ; dvc pull" {
  dvc remote add -d store "$remote" 2>&1 | Out-Host
  dvc push 2>&1 | Select-Object -Last 4
  Write-Host ""
  Write-Host "files now in the remote:"
  (Get-ChildItem -Recurse -File $remote).Count
  Write-Host ""
  Write-Host "deleting the local copy AND the local cache ..." -ForegroundColor Yellow
  Remove-Item readings.csv -Force
  Remove-Item -Recurse -Force .dvc\cache -ErrorAction SilentlyContinue
  Write-Host ("readings.csv exists : " + (Test-Path readings.csv))
  Write-Host ""
  Write-Host "dvc pull:"
  dvc pull 2>&1 | Select-Object -Last 5
  Write-Host ""
  Write-Host "recovered contents:" -ForegroundColor Green
  Get-Content readings.csv
}

Set-Location $root
Clear-Host
Write-Host ("=" * 76) -ForegroundColor Green
Write-Host "  DVC toy lab finished" -ForegroundColor Green
Write-Host ("=" * 76) -ForegroundColor Green
Write-Host ""
Write-Host ("unique screenshots : " + $script:seen.Count + " of " + $script:n)
Write-Host ""
foreach ($l in $script:man) { Write-Host ("  " + $l) }
$script:man | Out-File -Encoding utf8 (Join-Path $root "evidence\dvc_manifest.txt")
Write-Host ""
Write-Host ("saved to " + $shotDir)
Stop-Transcript | Out-Null
