# IL pipeline for one env v3 mode (detached): A* demos -> BC 1k/10k/50k -> DAgger. Usage: run_il_v3.ps1 -Mode static
param([string]$Mode)
Set-Location "E:\code\tum-adlr-09-main\tum-adlr-09-main\SAC"
$py = "D:\program\anaconda\envs\torch\python.exe"
$env:OMP_NUM_THREADS = "2"
$L = "..\results\v3\logs\il_$Mode.out"
& $py il_v3.py demos --mode $Mode --episodes 50000 *>> $L
foreach ($n in 1000, 10000, 50000) { & $py il_v3.py bc --mode $Mode --n-episodes $n --steps 20000 *>> $L }
& $py il_v3.py dagger --mode $Mode --init-episodes 1000 --init-steps 10000 --iters 10 --eps-per-iter 1000 --iter-steps 4000 *>> $L
Add-Content $L "IL PIPELINE DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
