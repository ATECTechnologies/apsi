# Apsi GPU Tuner for Windows (PowerShell part). $Mode is 'apply' or 'restore'.
$ErrorActionPreference = 'Continue'
$TaskName = 'Apsi GPU Tuner'
$Dir = Join-Path $env:ProgramData 'Apsi\GPU-Tuner'
function Say($t) { Write-Host $t }
$smi = @((Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'),
         (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $smi) { Say 'No NVIDIA driver found (nvidia-smi.exe is missing). Install the NVIDIA display driver first.'; return }

function Num($s) { $v = 0.0; if ([double]::TryParse(($s -replace '[^0-9.]', ''), [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$v)) { $v } else { $null } }
function Cards {
  & $smi --query-gpu=index,name,power.default_limit,power.min_limit,power.max_limit,power.limit,temperature.gpu --format=csv,noheader,nounits |
    ForEach-Object {
      $f = $_ -split ',\s*'
      [pscustomobject]@{ Index = [int]$f[0]; Name = $f[1].Trim(); Default = Num $f[2]; Min = Num $f[3]; Max = Num $f[4]; Now = Num $f[5]; Temp = $f[6] }
    }
}
# The recommended sustained-training power limit, as a share of the card's default limit.
function Recommend($name) {
  $n = $name.ToUpper()
  if ($n -match 'LAPTOP|MOBILE|MAX-Q') { return @(0, 'laptop GPU: the power limit is managed by the laptop, not adjustable') }
  if ($n -match '\b(H100|H200|H800|H20|GH200|B100|B200|GB200|B300|A100|A800|A30|A40|A10|A16|A2|L4|L40S?|T4|V100|P100|P40)\b|TESLA') {
    return @(100, 'data-center GPU: built and cooled for full sustained power; keep the default') }
  if ($n -match 'RTX PRO|RTX A\d|RTX \d+ ADA|QUADRO|RTX 6000|RTX 5000|RTX 4500|RTX 4000') { return @(90, 'workstation GPU: blower cooling, -10 % power costs about 2 % speed') }
  if ($n -match 'RTX 50\d\d') { return @(75, 'GeForce RTX 50: about -25 % power for about -5 % training speed') }
  if ($n -match 'RTX 40\d\d') { return @(75, 'GeForce RTX 40: about -25 % power for about -5 % training speed') }
  if ($n -match 'RTX 30\d\d') { return @(80, 'GeForce RTX 30: about -20 % power for about -5 % training speed, 6-10 C cooler') }
  if ($n -match 'RTX 20\d\d|TITAN RTX|GTX 16\d\d|GTX 10\d\d|TITAN') { return @(85, 'GeForce RTX 20 / GTX 16 / GTX 10: -15 % power for about -4 % speed') }
  return @(85, 'other NVIDIA GPU: a moderate -15 %')
}
$py = Join-Path $env:LOCALAPPDATA 'Programs\Apsi\runtime\python\python.exe'
function ApsiHelper($what) {
  if (-not (Test-Path $py)) { return }
  & $py -c "import apsi.gpu_helper" 2>$null
  if ($LASTEXITCODE -ne 0) { return }
  Say "Apsi is installed: apsi gpu $what"
  & $py -X utf8 -m apsi gpu $what
}

Say ''
Say '  Apsi GPU Tuner  -  best sustained settings for AI training'
Say '  -------------------------------------------------------'
$cards = @(Cards)
foreach ($c in $cards) { Say ("  GPU {0}: {1}  |  {2} C  |  power limit {3} W (default {4} W, allowed {5}-{6} W)" -f $c.Index, $c.Name, $c.Temp, $c.Now, $c.Default, $c.Min, $c.Max) }
Say ''

if ($Mode -eq 'apply') {
  $lines = @()
  foreach ($c in $cards) {
    $r = Recommend $c.Name
    if ($r[0] -eq 0 -or $c.Default -eq $null) { Say ("  GPU {0}: skipped ({1})" -f $c.Index, $r[1]); continue }
    $w = [math]::Round([math]::Max($c.Min, $c.Default * $r[0] / 100.0))
    Say ("  GPU {0}: {1} % of default = {2} W  ({3})" -f $c.Index, $r[0], $w, $r[1])
    & $smi -i $c.Index -pl $w | Out-Null
    if ($LASTEXITCODE -ne 0) { Say "           the driver refused the limit (this card or driver does not allow it)" ; continue }
    $lines += ('"{0}" -i {1} -pl {2}' -f $smi, $c.Index, $w)
  }
  if ($lines.Count) {
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    $cmd = Join-Path $Dir 'apply-at-start.cmd'
    Set-Content -Path $cmd -Encoding ASCII -Value (@('@echo off', 'rem Written by the Apsi GPU Tuner; undo with its Restore file.') + $lines)
    $act = New-ScheduledTaskAction -Execute $cmd
    $trg = @((New-ScheduledTaskTrigger -AtStartup), (New-ScheduledTaskTrigger -AtLogOn))
    $pri = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    $set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
    Register-ScheduledTask -TaskName $TaskName -Action $act -Trigger $trg -Principal $pri -Settings $set -Force | Out-Null
    Say ''
    Say "  Kept after every restart and logon (Task Scheduler: '$TaskName')."
  }
  ApsiHelper 'setup'
} else {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  if (Test-Path $Dir) { Remove-Item -Recurse -Force $Dir }
  foreach ($c in $cards) {
    if ($c.Default -eq $null) { continue }
    & $smi -i $c.Index -pl ([math]::Round($c.Default)) | Out-Null
    Say ("  GPU {0}: back to the default {1} W" -f $c.Index, [math]::Round($c.Default))
  }
  ApsiHelper 'remove'
  Say ''
  Say '  Everything the tuner set is removed; the cards run at their factory defaults.'
}
Say ''
foreach ($c in @(Cards)) { Say ("  now GPU {0}: power limit {1} W, {2} C" -f $c.Index, $c.Now, $c.Temp) }
