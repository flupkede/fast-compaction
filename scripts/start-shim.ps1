# Start (or restart) the fast-compaction shim. Safe to run any time:
# kills whatever listens on the port first, starts hidden, verifies.
# Writes its steps to %TEMP%\shim-start.log so invisible launches stay debuggable.
$ErrorActionPreference = 'Continue'
$repo = 'C:\WorkArea\AI\fast-compaction'
$port = 11435
$npx = 'C:\nvm4w\nodejs\npx.cmd'
$log = "$env:TEMP\shim-start.log"
function Step($text) { Add-Content -Path $log -Value "$(Get-Date -Format 'HH:mm:ss') $text" }

Step "run start"
$existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $existing) {
  Step "kill listener pid $($conn.OwningProcess)"
  Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
}

Step "spawn npx"
Start-Process -FilePath $npx `
  -ArgumentList 'tsx','scripts\systemone-shim.ts',"--port","$port",'--ollama','http://localhost:11434','--model','fast-compaction','--num-ctx','24576','--keep-alive','8760h' `
  -WorkingDirectory $repo -WindowStyle Hidden `
  -RedirectStandardOutput "$env:TEMP\shim.log" -RedirectStandardError "$env:TEMP\shim.err.log"

$ok = $false
for ($i = 0; $i -lt 25; $i++) {
  Start-Sleep -Milliseconds 800
  try {
    $r = Invoke-RestMethod -Uri "http://localhost:$port/v1/systemone" -Method Post `
      -ContentType 'application/json' -Body '{"state":"ping","questions":{"q1":{"type":"noul","instructions":"is this a ping?"}}}'
    Step "verified up (q1=$($r.answers.q1.noul))"
    $ok = $true
    break
  } catch { }
}
if (-not $ok) { Step "FAILED to come up; see shim.err.log" }
