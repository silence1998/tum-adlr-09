$py = "D:\program\anaconda\envs\torch\python.exe"
Set-Location "E:\code\tum-adlr-09-main\tum-adlr-09-main\SAC"
$log = "..\results\v3\logs\il_static_ablation.out"
"start $(Get-Date)" | Out-File $log -Encoding utf8
& $py il_v3.py bc --mode static --n-episodes 50000 --steps 100000 --hidden 1024 --suffix _h1024 *>> $log
& $py il_v3.py dagger --mode static --hidden 1024 --init-steps 20000 --iter-steps 10000 --suffix _h1024 *>> $log
"ABLATION DONE $(Get-Date)" | Out-File $log -Append -Encoding utf8
