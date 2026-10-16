# Start (or restart) the fast-compaction shim. Safe to run any time:
# kills whatever listens on the port first, starts hidden, verifies.
$ErrorActionPreference = 'Stop'
$repo = 'C:\WorkArea\AI\fast-compaction'
$port = 11435

$existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $existing) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }

Start-Process -FilePath 'npx.cmd' `
  -ArgumentList 'tsx','scripts\systemone-shim.ts',"--port","$port",'--ollama','http://localhost:11434','--model','fast-compaction','--num-ctx','16384','--keep-alive','1h' `
  -WorkingDirectory $repo -WindowStyle Hidden `
  -RedirectStandardOutput "$env:TEMP\shim.log" -RedirectStandardError "$env:TEMP\shim.err.log"

$ok = $false
for ($i = 0; $i -lt 20; $i++) {
  Start-Sleep -Milliseconds 800
  try {
    $r = Invoke-RestMethod -Uri "http://localhost:$port/v1/systemone" -Method Post `
      -ContentType 'application/json' -Body '{"state":"ping","questions":{"q1":{"type":"noul","instructions":"is this a ping?"}}}'
    Write-Output "shim up on :$port (judge answered q1=$($r.answers.q1.noul))"
    $ok = $true
    break
  } catch { }
}
if (-not $ok) { Write-Output "shim did NOT come up; see $env:TEMP\shim.err.log"; exit 1 }
