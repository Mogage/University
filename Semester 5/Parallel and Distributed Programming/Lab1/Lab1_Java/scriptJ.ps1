
$param1 = $args[0] # Nume fisier java
#Write-Host $param1

$param2 = $args[1] # No of threads
#Write-Host $param2

$param3 = $args[2] # Run type
#Write-Host $param2

$param4 = $args[3] # No of runs
#Write-Host $param2

# Executare class Java

$suma = 0

for ($i = 0; $i -lt $param4; $i++){
    Write-Host "Rulare" ($i+1)
    $a = C:\Users\nicol\.jdks\openjdk-21\bin\java.exe "-javaagent:D:\IntelliJ IDEA 2023.2.3\lib\idea_rt.jar=58958:D:\IntelliJ IDEA 2023.2.3\bin" -classpath "D:\1. Repositories\University\Semester 5\Parallel and Distributed Programming\Lab1\Lab1_Java\out\production\Lab1_Java" $args[0] $args[1] $args[2]# rulare class java
    Write-Host $a
    $suma += $a
    Write-Host ""
}
$media = $suma / $i
#Write-Host $suma
Write-Host "Timp de executie mediu:" $media

# Creare fisier .csv
if (!(Test-Path outJ.csv)){
    New-Item outJ.csv -ItemType File
    #Scrie date in csv
    Set-Content outJ.csv 'Tip Matrice,Nr threads,Timp executie'
}

# Append
Add-Content outJ.csv ",$($args[1]),$($media)"