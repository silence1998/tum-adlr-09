# Detached PPO runs on env v3. Usage: run_v3_ppo.ps1 -Jobs "static:0","random:0","bounce:0" [-Steps 3000000] [-Tag ""]
param([string[]]$Jobs, [int]$Steps = 3000000, [string]$Tag = "")
Set-Location "E:\code\tum-adlr-09-main\tum-adlr-09-main\SAC"
$py = "D:\program\anaconda\envs\torch\python.exe"
$R = "..\results\v3"
New-Item -ItemType Directory "$R\runs", "$R\logs", "$R\checkpoints" -Force | Out-Null
$env:OMP_NUM_THREADS = "2"
foreach ($j in $Jobs) {
  $mode, $seed = $j.Split(":")
  $name = "ppo_$mode$Tag" + "_s$seed"
  $cfgf = "..\configs\v3_$mode$Tag.yaml"
  $a = @("train_ppo_v3.py", "--env-config", $cfgf, "--total-steps", "$Steps", "--seed", $seed,
         "--log", "$R\runs\$name.csv", "--ckpt", "$R\checkpoints\$name.pt")
  $p = Start-Process -FilePath $py -ArgumentList $a -WindowStyle Hidden -PassThru `
       -RedirectStandardOutput "$R\logs\$name.out" -RedirectStandardError "$R\logs\$name.err"
  Add-Content "$R\logs\pids.log" "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') start $name pid $($p.Id) cfg $cfgf steps $Steps"
}
