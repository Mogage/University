<?php
$connection = new PDO("sqlite:../database.db");

$sql = "SELECT * FROM prob2";

$stmt = $connection->prepare($sql);
$stmt->execute();

$users = $stmt->fetchAll();

$index = -1;
$startIndex = $_GET['startIndex'];
$finalIndex = $startIndex + 3;

foreach ($users as $user) {
    $index++;
    if ($index < $startIndex) {
        continue;
    }
    if ($index >= $finalIndex) {
        break;
    }
    echo "<tr>";
    echo "<td>" . $user['nume'] . "</td>";
    echo "<td>" . $user['prenume'] . "</td>";
    echo "<td>" . $user['telefon'] . "</td>";
    echo "<td>" . $user['email'] . "</td>";
    echo "</tr>";
}

?>
