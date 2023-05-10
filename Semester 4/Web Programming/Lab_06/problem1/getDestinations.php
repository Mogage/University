<?php
$connection = new PDO("sqlite:../database.db");

$sql = "SELECT destinatie FROM Trenuri WHERE plecare=:plecare";

$stmt = $connection->prepare($sql);
$stmt->bindValue(':plecare', $_GET['q']);
$stmt->execute();

$destinations = $stmt->fetchAll();

foreach ($destinations as $destination) {
    echo "<option value='" . $destination['destinatie'] . "'>" . $destination['destinatie'] . "</option>";
}

?>
