
$param1 = $args[0] # Nume fisier exe
$param2 = $args[1] # Run type
$param3 = $args[2] # No of readers
$param4 = $args[3] # No of workers
$param5 = $args[4] # No of runs

# Executare exe in cmd mode

$suma = 0

for ($i = 0; $i -lt $param5; $i++){
    Write-Host "Rulare" ($i+1)
    $a = (cmd /c .\$param1 2 $param2 $param3 $param4 2`>`&1)
    Write-Host $a
    $suma += $a
    Write-Host ""
}
$media = $suma / $i
#Write-Host $suma
Write-Host "Timp de executie mediu:" $media

# Creare fisier .csv
if (!(Test-Path outC.csv)){
    New-Item outC.csv -ItemType File
    #Scrie date in csv
    Set-Content outC.csv 'Nr Readers, Nr Workers ,Timp executie'
}

# Append
Add-Content outC.csv "$($args[2]),$($args[3]),$($media)"