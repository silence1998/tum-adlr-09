# ARCHIVED 2026-10-03 (moved to SAC/legacy/): BROKEN - targets results/new_env, which no longer exists. Do not use.
# sequential queue: one training process at a time (GPU shared with the user's game)
param([int]$WaitPid = 0)
$ErrorActionPreference = "Continue"
Set-Location "E:\code\tum-adlr-09-main\tum-adlr-09-main\SAC"
$py = "D:\program\anaconda\envs\torch\python.exe"
if ($WaitPid -gt 0) { while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep 5 } }
$sacx = @("training_SAC-X.py","--scheduler","Q","--num-envs","8","--updates-per-step","1","--batch","32",
          "--features","relative_sorted","--main-reward-scale","1.0","--alpha-init","0.05","--warmup","5000",
          "--aux-reward-scale","0.1","--total-steps","300000","--mem-frac","0.3")
$jobs = @(
  @{n="sac_s1";   a=@("training_SAC.py","--seed","1","--total-steps","300000","--mem-frac","0.3")},
  @{n="sac_s2";   a=@("training_SAC.py","--seed","2","--total-steps","300000","--mem-frac","0.3")},
  @{n="sacx_q_s0"; a=$sacx+@("--seed","0")},
  @{n="sacx_q_s1"; a=$sacx+@("--seed","1")},
  @{n="sacx_q_s2"; a=$sacx+@("--seed","2")},
  @{n="ppo_sorted_s0"; a=@("training_PPO.py","--features","relative_sorted","--seed","0","--total-steps","1000000","--mem-frac","0.3")},
  @{n="ppo_sorted_s1"; a=@("training_PPO.py","--features","relative_sorted","--seed","1","--total-steps","1000000","--mem-frac","0.3")},
  @{n="ppo_sorted_s2"; a=@("training_PPO.py","--features","relative_sorted","--seed","2","--total-steps","1000000","--mem-frac","0.3")}
)
foreach ($j in $jobs) {
  if (Test-Path "..\results\new_env\queue_skip_$($j.n)") { continue }
  $args2 = $j.a + @("--log", "..\results\new_env\runs\$($j.n).csv")
  $t0 = Get-Date
  $p = Start-Process -FilePath $py -ArgumentList $args2 -WindowStyle Hidden -PassThru `
       -RedirectStandardOutput "..\results\new_env\logs\$($j.n)_fixed.out" -RedirectStandardError "..\results\new_env\logs\$($j.n)_fixed.err"
  Add-Content "..\results\new_env\logs\queue.log" "$(Get-Date -Format HH:mm:ss) start $($j.n) pid $($p.Id)"
  $p.WaitForExit()
  $dt = ((Get-Date) - $t0).TotalSeconds
  Add-Content "..\results\new_env\logs\queue.log" "$(Get-Date -Format HH:mm:ss) end $($j.n) exit $($p.ExitCode) wall $([int]$dt)s"
}
Add-Content "..\results\new_env\logs\queue.log" "$(Get-Date -Format HH:mm:ss) queue done"
