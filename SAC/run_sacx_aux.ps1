# Parallel SAC-X aux-task ablation launcher (results/sacx_aux/). Usage: run_sacx_aux.ps1 -Batch 1|2
param([int]$Batch = 1)
Set-Location "E:\code\tum-adlr-09-main\tum-adlr-09-main\SAC"
$py = "D:\program\anaconda\envs\torch\python.exe"
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$R = "..\results\sacx_aux"
$common = @("training_SAC-X.py","--num-envs","8","--updates-per-step","1","--batch","32","--features","relative_sorted",
            "--main-reward-scale","1.0","--alpha-init","0.05","--warmup","5000","--total-steps","1000000",
            "--ckpt-dir","$R\checkpoints")
$VARS = @{
  "aux_main"       = @("--tasks","0","--aux-reward-scale","0.1")
  "aux_approach"   = @("--tasks","0,1","--aux-reward-scale","0.1")
  "aux_axis"       = @("--tasks","0,2,3,4,5","--aux-reward-scale","0.1")
  "aux_obsaware"   = @("--tasks","0,6,7","--aux-reward-scale","0.1")
  "aux_full_U"     = @("--tasks","0,1,2,3,4,5","--scheduler","U","--aux-reward-scale","0.1")
  "aux_full_xi100" = @("--tasks","0,1,2,3,4,5","--xi","100","--aux-reward-scale","0.1")
  "aux_full_scale1"= @("--tasks","0,1,2,3,4,5","--aux-reward-scale","1.0")
  "aux_full_seq1"  = @("--tasks","0,1,2,3,4,5","--seq-len","1","--batch","256","--aux-reward-scale","0.1")
  "aux_appr_obs"   = @("--tasks","0,1,6,7","--aux-reward-scale","0.1")
}
$jobs = Get-Content "$R\batch$Batch.txt" | ? { $_.Trim() -ne "" }   # lines: <variant> <seed>
foreach ($line in $jobs) {
  $v, $s = $line.Split(" ")
  $name = "${v}_s$s"
  $a = $common + $VARS[$v] + @("--seed", $s, "--log", "$R\runs\$name.csv")
  $p = Start-Process -FilePath $py -ArgumentList $a -WindowStyle Hidden -PassThru `
       -RedirectStandardOutput "$R\logs\$name.out" -RedirectStandardError "$R\logs\$name.err"
  Add-Content "$R\logs\pids.log" "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') batch $Batch start $name pid $($p.Id)"
}
