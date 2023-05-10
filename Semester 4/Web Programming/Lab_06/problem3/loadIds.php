<?php

$connection = new PDO("sqlite:../database.db");

$sql = "SELECT id FROM prob3";

$stmt = $connection->prepare($sql);
$stmt->execute();

$ids= $stmt->fetchAll();

foreach ($ids as $id) {
    echo "<li onclick=\"loadForm(" . $id['id'] . ")\">" . $id['id'] . "</li>";
}

?>