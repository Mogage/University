$param1 = $args[0] # Nume fisier exe
$param2 = $args[1] # No of threads
$param3 = $args[2] # No of runs

# Executare exe in cmd mode

$sumaT1 = 0
$sumaT2 = 0

for ($i = 0; $i -lt $param3; $i++){
    Write-Host "Rulare" ($i+1)
    # $a = (cmd /c .\$param1 2 $param2 $param3 2`>`&1)
    $output = mpiexec -n $param2 $param1 2  2>&1

    $lines = $output -split "`n"  
    $valueT1 = [double]($lines[0])
    $valueT2 = [double]($lines[1])

    $sumaT1 += $valueT1
    $sumaT2 += $valueT2

    Write-Host "T1 for run $($i+1): $valueT1"
    Write-Host "T2 for run $($i+1): $valueT2"
    Write-Host ""
}

$mediaT1 = $sumaT1 / $i
$mediaT2 = $sumaT2 / $i

#Write-Host
Write-Host "Timp de executie mediu pentru T1:" $mediaT1
Write-Host "Timp de executie mediu pentru T2:" $mediaT2


# Creare fisier .csv
if (!(Test-Path outC.csv)){
    New-Item outC.csv -ItemType File
    #Scrie date in csv
    Set-Content outC.csv 'Tip Executie, Nr procese, Timp executie T1, Timp executie T2'
}

# Append
Add-Content outC.csv "$param4,$($args[1]),$mediaT1,$mediaT2"
# Add-Content outC.csv ",,$($args[1]),$($media)"