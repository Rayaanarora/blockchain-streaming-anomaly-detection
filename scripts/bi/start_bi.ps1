# Start the BI layer (Superset web on :8088 and Superset's MCP service on :5008) inside Ubuntu WSL on D:.
# Runs them in hidden background processes that keep running after this window closes.
#   powershell -ExecutionPolicy Bypass -File scripts\bi\start_bi.ps1          # start
#   powershell -ExecutionPolicy Bypass -File scripts\bi\start_bi.ps1 -Stop    # stop
param([switch]$Stop)
$repo = "/mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection"
$log = "D:\blockchainbigdata\blockchain-streaming-anomaly-detection\data\bi\logs"
New-Item -ItemType Directory -Force $log | Out-Null

if ($Stop) {
    wsl -d Ubuntu -- bash "$repo/scripts/bi/stop_bi.sh"
    return
}
function Test-Port($url) { try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 $url).StatusCode -eq 200 } catch { $false } }

if (-not (Test-Port "http://127.0.0.1:8088/health")) {
    Start-Process -WindowStyle Hidden -FilePath wsl.exe -ArgumentList "-d Ubuntu -- bash $repo/scripts/bi/start_superset.sh --log"
}
Start-Sleep -Seconds 2
$mcpUp = $false
try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Method Post -Uri "http://127.0.0.1:5008/mcp" -ContentType "application/json" -Headers @{Accept = "application/json, text/event-stream"} -Body '{"jsonrpc":"2.0","id":1,"method":"ping"}'; $mcpUp = $true } catch { if ($_.Exception.Response) { $mcpUp = $true } }
if (-not $mcpUp) {
    Start-Process -WindowStyle Hidden -FilePath wsl.exe -ArgumentList "-d Ubuntu -- bash $repo/scripts/bi/start_superset_mcp.sh --log"
}
Write-Host "waiting for Superset ..."
for ($i = 0; $i -lt 60; $i++) { if (Test-Port "http://127.0.0.1:8088/health") { break }; Start-Sleep 3 }
if (Test-Port "http://127.0.0.1:8088/health") { Write-Host "Superset: http://127.0.0.1:8088   MCP: http://127.0.0.1:5008/mcp" }
else { Write-Host "Superset did not start; see $log\superset.log" }
