$n_runs = @(10, 10, 3)
$suma = 0
$narray = @(10, 1000, 10000)
$nthreads = @(0, 2, 4, 8, 16)
# Creare fisier .csv
if (!(Test-Path outC.csv)){
    New-Item outC.csv -ItemType File
    #Scrie date in csv
    Set-Content outC.csv 'Tip Matrice,Nr threads,Timp executie'
}

for ($j = 0; $j -lt $narray.Length; $j++){
	$n = $($narray[$j])
	for ($k = 0; $k -lt $nthreads.Length; $k++){
		$threads = $($nthreads[$k])
		$suma = 0
		for ($i = 0; $i -lt $n_runs[$j]; $i++){
			$time = & "x64\Debug\P2_21.exe" $threads $n $n "3" 2>&1
			$suma += [double]$time
			Write-Host "Rulare $i -> $time"
		}
		$media = $suma / $n_runs[$j]
		Write-Host "[N=M=$n] [THREADS=$threads] => " $media
		Add-Content outC.csv "$n,$threads,$media"
	}
}
