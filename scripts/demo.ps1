<#
.SYNOPSIS
    StreamGuard end-to-end demo: brings the stack up, drives a 5000-event
    fraud burst through RabbitMQ, exercises every API surface, retrains,
    and finishes with the offline evaluation (real precision/recall/F1).

.EXAMPLE
    .\scripts\demo.ps1                # up (rebuild if needed) -> burst -> checks -> evaluate
    .\scripts\demo.ps1 -Fresh         # docker compose down -v first (clean slate)
    .\scripts\demo.ps1 -BurstCount 2000

    Exits 0 when every check passes, 1 otherwise.
#>
param(
    [int]$BurstCount = 5000,
    [switch]$Fresh,
    [int]$ConsumeTimeoutSec = 180,
    [int]$SoakSeconds = 120
)

$ErrorActionPreference = "Continue"
$api = "http://localhost:3300"
$checks = [System.Collections.ArrayList]::new()

function Step($msg) { Write-Host "`n=== $msg" -ForegroundColor Cyan }
function Pass($name, $detail = "") {
    [void]$checks.Add(@{ name = $name; ok = $true; detail = $detail })
    Write-Host "  [PASS] $name $detail" -ForegroundColor Green
}
function Fail($name, $detail = "") {
    [void]$checks.Add(@{ name = $name; ok = $false; detail = $detail })
    Write-Host "  [FAIL] $name $detail" -ForegroundColor Red
}
function Check($name, $cond, $detail = "") {
    if ($cond) { Pass $name $detail } else { Fail $name $detail }
}
function Invoke-Json($url, $Method = "GET", $Body = $null, $TimeoutSec = 30) {
    $params = @{ Uri = $url; Method = $Method; UseBasicParsing = $true; TimeoutSec = $TimeoutSec }
    if ($Body) {
        $params.Body = ($Body | ConvertTo-Json -Compress)
        $params.ContentType = "application/json"
    }
    $resp = Invoke-WebRequest @params
    return $resp.Content | ConvertFrom-Json
}

# --------------------------------------------------------------------------
Step "0/8 Clean slate (optional)"
if ($Fresh) {
    Write-Host "  down -v ..."
    cmd /c "docker compose down -v >nul 2>&1"
    Check "docker compose down -v" ($LASTEXITCODE -eq 0)
}

# --------------------------------------------------------------------------
Step "1/8 Build + start the stack"
# cmd /c merges docker's stderr at process level (avoids PS 5.1 NativeCommandError)
cmd /c "docker compose up -d --build 2>&1" | Select-Object -Last 5
if ($LASTEXITCODE -ne 0) { Fail "docker compose up" "exit $LASTEXITCODE" } else { Pass "docker compose up" }

$deadline = (Get-Date).AddSeconds(240)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $h = Invoke-Json "$api/healthz/ready" -TimeoutSec 5
        if ($h.ready) { $ready = $true; break }
    } catch { }
    Start-Sleep -Seconds 3
}
Check "API ready (/healthz/ready)" $ready
if (-not $ready) { Write-Host "stack did not become ready; aborting" -ForegroundColor Red; exit 1 }

$metrics0 = Invoke-Json "$api/api/metrics"
Pass "baseline metrics" "eventsProcessed=$($metrics0.eventsProcessed) threshold=$($metrics0.threshold)"
$baseline = [int]$metrics0.eventsProcessed

# --------------------------------------------------------------------------
Step "2/8 Burst $BurstCount events (fast mode, disjoint user pool)"
cmd /c "docker compose exec -T simulator python -m app.simulator --count $BurstCount --fast 2>&1" | Select-Object -Last 3
Check "burst command exit 0" ($LASTEXITCODE -eq 0)

# --------------------------------------------------------------------------
Step "3/8 Wait for the consumer to drain the burst"
$deadline = (Get-Date).AddSeconds($ConsumeTimeoutSec)
$consumed = $false
$processed = 0
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    try {
        $m = Invoke-Json "$api/api/metrics" -TimeoutSec 10
        $processed = [int]$m.eventsProcessed
        if ($processed -ge $baseline + $BurstCount) { $consumed = $true; break }
        Write-Host "  consumed $processed / $($baseline + $BurstCount)"
    } catch { Write-Host "  poll failed (retrying): $($_.Exception.Message)" }
}
Check "burst fully consumed" $consumed "processed=$processed"

# Soak: let the steady 20 eps simulator keep running — this is where the
# hard-negative classes (power users, benign large purchases) show up, so
# the final evaluation is representative and not burst-pool-only.
if ($SoakSeconds -gt 0) {
    Write-Host "  soaking $SoakSeconds s of steady traffic ..."
    Start-Sleep -Seconds $SoakSeconds
}
Start-Sleep -Seconds 8   # settle: final batch inserts + alert inserts

# --------------------------------------------------------------------------
Step "4/8 Alerts API"
$alerts = Invoke-Json "$api/api/alerts?limit=5"
Check "alerts returned" ($alerts.count -gt 0) "count=$($alerts.count)"
if ($alerts.count -gt 0) {
    $a = $alerts.alerts[0]
    Write-Host ($alerts.alerts[0] | ConvertTo-Json -Depth 5)
    Check "alert has reasons[]" ($a.reasons.Count -gt 0) `
        "reasons=$((@($a.reasons | ForEach-Object { $_.rule })) -join ',')"
    Check "alert has detectionDelayMs" ($null -ne $a.detectionDelayMs) "delay=$($a.detectionDelayMs)ms"
    Check "alert risk in [0,1]" ($a.risk -ge 0 -and $a.risk -le 1) "risk=$($a.risk)"
}

$m = Invoke-Json "$api/api/metrics"
Check "metrics endpoint sane" ($m.eventsProcessed -gt 0 -and $m.alertsCount -gt 0) `
    "processed=$($m.eventsProcessed) alerts=$($m.alertsCount) avgDelay=$($m.avgDetectionDelayMs)ms p95=$($m.p95DetectionDelayMs)ms"

# --------------------------------------------------------------------------
Step "5/8 Feedback API"
$alertId = $alerts.alerts[0].id
try {
    $fb = Invoke-Json "$api/api/alerts/$alertId/feedback" -Method POST -Body @{ label = "true" }
    Check "feedback recorded" ($fb.recorded -eq $true) "alertId=$alertId label=$($fb.label)"
} catch {
    Fail "feedback recorded" $_.Exception.Message
}

# --------------------------------------------------------------------------
Step "6/8 SSE stream (/api/stream)"
$sseFile = Join-Path $env:TEMP "sg_sse.txt"
if (Test-Path $sseFile) { Remove-Item $sseFile }
# curl.exe (not the broken curl alias); --max-time bounds the infinite stream
curl.exe --max-time 6 -N "$api/api/stream?window_seconds=60" --output $sseFile 2>$null | Out-Null
$sse = ""
if (Test-Path $sseFile) { $sse = Get-Content $sseFile -Raw }
Check "SSE emits metrics frames" ($sse -match 'event: metrics') ""
Check "SSE payload is JSON metrics" ($sse -match '"eventsProcessed"') ""

# --------------------------------------------------------------------------
Step "7/8 Dashboard (static UI at /)"
$html = (Invoke-WebRequest "$api/" -UseBasicParsing -TimeoutSec 10).Content
Check "dashboard HTML served" ($html -match "StreamGuard") ""
Check "dashboard wires SSE + KPIs" ($html -match "kpi-events" -and $html -match "app\.js") ""
$js = (Invoke-WebRequest "$api/app.js" -UseBasicParsing -TimeoutSec 10).Content
Check "dashboard JS uses EventSource" ($js -match "EventSource") ""

# --------------------------------------------------------------------------
Step "8/8 Retrain + offline evaluation"
$v0 = Invoke-Json "$api/api/model/metrics"
Write-Host "  current model: v$($v0.modelVersion)"
try {
    $rt = Invoke-Json "$api/api/retrain" -Method POST -TimeoutSec 600
    Check "retrain returns new model" ($rt.modelVersion -gt $v0.modelVersion) `
        "v$($v0.modelVersion) -> v$($rt.modelVersion) primary=$($rt.primaryModel) liveRows=$($rt.liveRowsUsed) in $($rt.durationSeconds)s"
} catch {
    Fail "retrain returns new model" $_.Exception.Message
}

# cmd redirect keeps Python's UTF-8 stdout byte-for-byte (PS Tee-Object would
# write UTF-16 / BOM, which strict JSON parsers reject)
cmd /c "docker compose exec -T api python scripts/evaluate.py > docs\last-eval.json 2>&1"
Get-Content "$PSScriptRoot\..\docs\last-eval.json" -Raw | Write-Host
Check "evaluate.py exit 0" ($LASTEXITCODE -eq 0)

# --------------------------------------------------------------------------
Step "Summary"
$failed = @($checks | Where-Object { -not $_.ok })
Write-Host ("  {0}/{1} checks passed" -f ($checks.Count - $failed.Count), $checks.Count)
foreach ($f in $failed) { Write-Host "  FAILED: $($f.name) $($f.detail)" -ForegroundColor Red }
if ($failed.Count -eq 0) {
    Write-Host "`nDEMO PASS" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`nDEMO FAIL" -ForegroundColor Red
    exit 1
}
