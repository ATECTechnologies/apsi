@echo off
rem Uploads Apsi 2.6.0 (installers, the source code and the GPU Tuner) from your Downloads folder to apsi.dev.
rem One file: the PowerShell below this header does the work. Asks for an API token (Settings -> Developer,
rem "release"). Files already on the site are skipped, so running it again is safe.
set "APSI_UPLOAD_SELF=%~f0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = (Get-Content -LiteralPath $env:APSI_UPLOAD_SELF -Raw) -split '##[ ]PS[ ]##'; & ([scriptblock]::Create($s[1]))"
echo.
pause
exit /b
## PS ##
# Upload every Apsi build found in your Downloads folder to the website, in one go.
#
#   double-click upload-releases.bat (one file: this PowerShell runs from inside it)
#
# It uploads:
#   Downloads\apsi-X.Y.Z\installer\release\ApsiSetup-X.Y.Z-x64.exe   -> Windows installer
#   Downloads\apsi-X.Y.Z-linux.zip  (ApsiSetup-X.Y.Z-linux.sh and the .deb inside it)  -> Linux installer
#   source\apsi-X.Y.Z-source.zip in that zip                                           -> Source code
#   the Apsi GPU Tuner in that zip (gpu-tuner\: Apply / Restore for Windows and Linux) -> the Download page's
#   "Apsi GPU Tuner" section
# Versions: 2.6.0 and newer by default (-Version 2.6.0+); -Version 2.4.4 for one version only, -Version all for
# every version found. Files already on the site are skipped, so running it again is safe. Files are sent in 32 MB pieces, 4 at a time (Cloudflare accepts at most 100 MB per
# request); a broken piece is sent again.
#
# It asks for an API token once. Create it on the site: Settings -> Developer, tick "release", 30 days.
# The token is never written to disk.
param(
  [string]$Root = (Join-Path $HOME 'Downloads'),
  [string]$Site = 'https://apsi.dev',
  [string]$Version = '2.6.0+',
  [int]$Parallel = 4,
  [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.Net.Http

# ---------------------------------------------------------------- find the files
$items = New-Object System.Collections.ArrayList
function Add-Build($path, $version, $platform, $title) {
  [void]$items.Add([pscustomobject]@{ Path = $path; Version = $version; Platform = $platform; Title = $title; Name = (Split-Path $path -Leaf) })
}
$tmp = Join-Path $env:TEMP ('apsi-upload-' + [guid]::NewGuid().ToString('N').Substring(0, 8))

Get-ChildItem -LiteralPath $Root -Directory | ForEach-Object {
  if ($_.Name -notmatch '^apsi-(\d+\.\d+\.\d+)$') { return }
  $v = $Matches[1]
  Get-ChildItem -LiteralPath (Join-Path $_.FullName 'installer\release') -Filter 'ApsiSetup-*-x64.exe' -ErrorAction SilentlyContinue |
    ForEach-Object { Add-Build $_.FullName $v 'windows' 'Windows installer' }
}
Get-ChildItem -LiteralPath $Root -File -Filter 'apsi-*-linux.zip' | ForEach-Object {
  if ($_.Name -notmatch '^apsi-(\d+\.\d+\.\d+)-linux\.zip$') { return }
  $v = $Matches[1]
  $dir = Join-Path $tmp $v
  Expand-Archive -LiteralPath $_.FullName -DestinationPath $dir -Force
  Get-ChildItem -LiteralPath $dir -Recurse -File -Filter 'apsi-*-source.zip' | ForEach-Object { Add-Build $_.FullName $v 'source' 'Source code' }
  Get-ChildItem -LiteralPath $dir -Recurse -File | Where-Object { -not $_.FullName.Contains('\source\') } | ForEach-Object {
    if ($_.Name -like 'ApsiSetup-*-linux.sh') { Add-Build $_.FullName $v 'linux' 'Linux installer' }
    elseif ($_.Name -like '*.deb') { Add-Build $_.FullName $v 'linux' 'Debian / Ubuntu package (.deb)' }
    elseif ($_.Name -eq 'Apsi-GPU-Tuner-Apply.bat') { Add-Build $_.FullName $v 'windows' 'GPU Tuner: Apply recommended settings (Windows)' }
    elseif ($_.Name -eq 'Apsi-GPU-Tuner-Restore.bat') { Add-Build $_.FullName $v 'windows' 'GPU Tuner: Restore factory defaults (Windows)' }
    elseif ($_.Name -eq 'apsi-gpu-tuner-apply-linux.sh') { Add-Build $_.FullName $v 'linux' 'GPU Tuner: Apply recommended settings (Linux)' }
    elseif ($_.Name -eq 'apsi-gpu-tuner-restore-linux.sh') { Add-Build $_.FullName $v 'linux' 'GPU Tuner: Restore factory defaults (Linux)' }
  }
}
if ($Version -match '^(\d+\.\d+\.\d+)\+$') {
  $min = [version]$Matches[1]
  $items = @($items | Where-Object { [version]$_.Version -ge $min })
} elseif ($Version -ne 'all') { $items = @($items | Where-Object { $_.Version -eq $Version }) }
if (-not $items.Count) { Write-Host "No Apsi $Version builds found in $Root"; exit 1 }
Write-Host "Found:" -ForegroundColor Cyan
$items | Sort-Object Version, Platform | ForEach-Object { '  {0,-8} {1,-8} {2}  ({3:N1} MB)' -f $_.Version, $_.Platform, $_.Name, ((Get-Item -LiteralPath $_.Path).Length / 1MB) }
if ($DryRun) { exit 0 }

# ---------------------------------------------------------------- talk to the site
$sec = Read-Host 'API token (Settings -> Developer, with "release")' -AsSecureString
$token = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
if (-not $token.StartsWith('apsi_')) { Write-Host 'That does not look like an Apsi token (they start with apsi_).' -ForegroundColor Red; exit 1 }

$http = New-Object System.Net.Http.HttpClient
$http.Timeout = [TimeSpan]::FromMinutes(15)
$http.DefaultRequestHeaders.Authorization = New-Object System.Net.Http.Headers.AuthenticationHeaderValue('Bearer', $token)
$http.DefaultRequestHeaders.UserAgent.ParseAdd('apsi-release-upload/1.0')

function Call($method, $path, $content) {
  $req = New-Object System.Net.Http.HttpRequestMessage((New-Object System.Net.Http.HttpMethod($method)), "$Site$path")
  if ($content) { $req.Content = $content }
  $res = $http.SendAsync($req).GetAwaiter().GetResult()
  $text = $res.Content.ReadAsStringAsync().GetAwaiter().GetResult()
  $data = $null
  try { $data = $text | ConvertFrom-Json } catch {}
  [pscustomobject]@{ Status = [int]$res.StatusCode; Data = $data; Text = $text }
}
function Json($obj) { New-Object System.Net.Http.StringContent(($obj | ConvertTo-Json -Compress), [Text.Encoding]::UTF8, 'application/json') }
function Fail($r, $what) {
  $msg = if ($r.Data -and $r.Data.error) { $r.Data.error } else { $r.Text.Substring(0, [Math]::Min(300, $r.Text.Length)) }
  throw "$what failed ($($r.Status)): $msg"
}

$r = Call 'GET' '/api/v1/releases?all=1' $null
if ($r.Status -ne 200) { Fail $r 'Checking the token' }
$have = @{}
foreach ($x in $r.Data.releases) { $have["$($x.version)|$($x.filename)"] = $true }

$done = 0; $skipped = 0; $failed = 0
foreach ($it in ($items | Sort-Object Version, Platform)) {
  if ($have["$($it.Version)|$($it.Name)"]) { Write-Host "skip  $($it.Version) $($it.Name) (already on the site)" -ForegroundColor DarkGray; $skipped++; continue }
  try {
    $size = (Get-Item -LiteralPath $it.Path).Length
    Write-Host "send  $($it.Version) $($it.Name)  (checking the file)" -NoNewline
    $sha = (Get-FileHash -LiteralPath $it.Path -Algorithm SHA256).Hash.ToLower()
    $r = Call 'POST' '/api/v1/uploads' (Json @{ purpose = 'release'; size = $size; sha256 = $sha; filename = $it.Name })
    if ($r.Status -ne 201) { Fail $r 'Starting the upload' }
    $id = $r.Data.id; $chunk = [int64]$r.Data.chunk_size
    $count = [int][Math]::Ceiling($size / $chunk)
    $queue = New-Object 'System.Collections.Generic.Queue[int]'
    for ($i = 0; $i -lt $count; $i++) { $queue.Enqueue($i) }
    $running = @{}; $tries = @{}; $sent = [int64]0; $t0 = Get-Date
    $fs = [IO.File]::OpenRead($it.Path)
    try {
      while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        while ($running.Count -lt $Parallel -and $queue.Count -gt 0) {
          $i = $queue.Dequeue()
          $off = [int64]$i * $chunk
          $len = [int][Math]::Min($chunk, $size - $off)
          $buf = New-Object byte[] $len
          $fs.Position = $off
          $n = 0
          while ($n -lt $len) { $k = $fs.Read($buf, $n, $len - $n); if ($k -le 0) { break }; $n += $k }
          $c = New-Object System.Net.Http.ByteArrayContent($buf, 0, $n)
          $c.Headers.ContentRange = New-Object System.Net.Http.Headers.ContentRangeHeaderValue($off, ($off + $n - 1), $size)
          $c.Headers.ContentType = New-Object System.Net.Http.Headers.MediaTypeHeaderValue('application/octet-stream')
          $req = New-Object System.Net.Http.HttpRequestMessage((New-Object System.Net.Http.HttpMethod('PUT')), "$Site/api/v1/uploads/$id")
          $req.Content = $c
          $running[$http.SendAsync($req)] = @{ Index = $i; Length = $n }
        }
        $tasks = [System.Threading.Tasks.Task[]]@($running.Keys)
        $w = [System.Threading.Tasks.Task]::WaitAny($tasks)
        $task = $tasks[$w]; $info = $running[$task]; $running.Remove($task)
        $ok = $false; $why = ''
        if ($task.Status -eq 'RanToCompletion') {
          $resp = $task.Result
          $ok = $resp.IsSuccessStatusCode
          if (-not $ok) { $why = "$([int]$resp.StatusCode) " + $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult() }
          $resp.Dispose()
        } elseif ($task.Exception) { $why = $task.Exception.GetBaseException().Message }
        if ($ok) {
          $sent += $info.Length
          $secs = [Math]::Max(0.1, ((Get-Date) - $t0).TotalSeconds)
          Write-Host ("`rsend  {0} {1}  {2,3}%  {3:N1} MB/s   " -f $it.Version, $it.Name, [int](100 * $sent / $size), ($sent / 1MB / $secs)) -NoNewline
        } else {
          $tries[$info.Index] = 1 + [int]$tries[$info.Index]
          if ($tries[$info.Index] -gt 6) { throw "Sending piece $($info.Index + 1) of $count failed: $why" }
          Start-Sleep -Seconds (2 * $tries[$info.Index])
          $queue.Enqueue($info.Index)
        }
      }
    } finally { $fs.Dispose() }
    $r = Call 'POST' '/api/v1/releases' (Json @{ upload_id = $id; version = $it.Version; platform = $it.Platform; title = $it.Title; filename = $it.Name; sha256 = $sha })
    if ($r.Status -ne 201) { Fail $r 'Publishing' }
    Write-Host ' done' -ForegroundColor Green
    $done++
  } catch {
    Write-Host ''
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Red
    $failed++
  }
}
$http.Dispose()
if (Test-Path $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
Write-Host ''
Write-Host "Uploaded $done, already there $skipped, failed $failed.  See $Site/download/all" -ForegroundColor Cyan
