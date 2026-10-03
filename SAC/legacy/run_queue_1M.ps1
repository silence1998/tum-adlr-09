# ARCHIVED 2026-10-03 (moved to SAC/legacy/): BROKEN - targets results/new_env, which no longer exists. Do not use.
# sequential 1M-step queue (one training process at a time; GPU shared with the user's game)
$ErrorActionPreference = "Continue"
Set-Location "E:\code\tum-adlr-09-main\tum-adlr-09-main\SAC"
$py = "D:\program\anaconda\envs\torch\python.exe"
$out = "..\results\new_env\runs_1M"; $logs = "..\results\new_env\logs_1M"
New-Item -ItemType Directory $out, $logs -Force | Out-Null
$sacx = @("training_SAC-X.py","--scheduler","Q","--num-envs","8","--updates-per-step","1","--batch","32",
          "--features","relative_sorted","--main-reward-scale","1.0","--alpha-init","0.05","--warmup","5000",
          "--aux-reward-scale","0.1","--total-steps","1000000","--mem-frac","0.3")
$sac = @("training_SAC.py","--total-steps","1000000","--mem-frac","0.3")
$jobs = @(
  @{n="sac_s0"; a=$sac+@("--seed","0")}, @{n="sacx_q_s0"; a=$sacx+@("--seed","0")},
  @{n="sac_s1"; a=$sac+@("--seed","1")}, @{n="sacx_q_s1"; a=$sacx+@("--seed","1")},
  @{n="sac_s2"; a=$sac+@("--seed","2")}, @{n="sacx_q_s2"; a=$sacx+@("--seed","2")}
)
foreach ($j in $jobs) {
  if (Test-Path "$out\skip_$($j.n)") { continue }
  $t0 = Get-Date
  $p = Start-Process -FilePath $py -ArgumentList ($j.a + @("--log", "$out\$($j.n).csv")) -WindowStyle Hidden -PassThru `
       -RedirectStandardOutput "$logs\$($j.n).out" -RedirectStandardError "$logs\$($j.n).err"
  Add-Content "$logs\queue.log" "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') start $($j.n) pid $($p.Id)"
  $p.WaitForExit()
  Add-Content "$logs\queue.log" "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') end $($j.n) wall $([int]((Get-Date) - $t0).TotalSeconds)s"
}
Add-Content "$logs\queue.log" "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') queue done"
