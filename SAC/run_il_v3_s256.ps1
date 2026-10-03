$py = "D:\program\anaconda\envs\torch\python.exe"
Set-Location "E:\code\tum-adlr-09-main\tum-adlr-09-main\SAC"
$IL = "..\results\v3\il"
foreach ($m in @("static", "random", "bounce")) {
  $log = "..\results\v3\logs\il_s256_$m.out"
  "start $(Get-Date)" | Out-File $log -Encoding utf8
  foreach ($n in @(1000, 10000)) {
    $k = $n / 1000
    if (-not (Test-Path "$IL\bc_${m}_${k}k_s100k.pt")) { & $py il_v3.py bc --mode $m --n-episodes $n --steps 100000 --suffix _s100k 2>&1 | Out-File $log -Append -Encoding utf8 }
  }
  & $py il_v3.py dagger --mode $m --init-episodes 10000 --init-steps 100000 --iters 8 --eps-per-iter 5000 --iter-steps 30000 --suffix _big 2>&1 | Out-File $log -Append -Encoding utf8
  if (-not (Test-Path "$IL\bc_${m}_50k_s300k.pt")) { & $py il_v3.py bc --mode $m --n-episodes 50000 --steps 300000 --suffix _s300k 2>&1 | Out-File $log -Append -Encoding utf8 }
  "MODE DONE $m $(Get-Date)" | Out-File $log -Append -Encoding utf8
}
