
$param1 = $args[0] # Nume fisier exe
$param2 = $args[1] # Run type
$param3 = $args[2] # No of runs
$param4 = $args[3] # No of readers
$param5 = $args[4] # No of workers

# Executare exe in cmd mode

$suma = 0

for ($i = 0; $i -lt $param3; $i++){
    Write-Host "Rulare" ($i+1)
    $a = (cmd /c .\$param1 2 $param2 $param4 $param5 2`>`&1)
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
Add-Content outC.csv "$($args[3]),$($args[4]),$($media)"