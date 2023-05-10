<?php

$connection = new PDO("sqlite:../database.db");

$sql = "SELECT * FROM prob3";

$stmt = $connection->prepare($sql);
$stmt->execute();

$users= $stmt->fetchAll();

$id = $_GET['q'];

foreach ($users as $user) {
    if ($user['id'] != $id) {
        continue;
    }
    echo "<input type=\"text\" id=\"id\" placeholder=\"id\" value=\"" . $user['id'] . "\">";
    echo "<input type=\"text\" id=\"nume\" placeholder=\"nume\" value=\"" . $user['nume'] . "\">";
    echo "<input type=\"text\" id=\"prenume\" placeholder=\"prenume\" value=\"" . $user['prenume'] . "\">";
    echo "<input type=\"text\" id=\"email\" placeholder=\"email\" value=\"". $user['email'] . "\">";
    echo "<input type=\"text\" id=\"oras\" placeholder=\"oras\" value=\"" . $user['oras'] . "\">";
    echo "<input type=\"text\" id=\"strada\" placeholder=\"strada\" value=\"" . $user['strada'] . "\">";
    echo "<input type=\"submit\" value=\"Submit\">";
    break;
}

?>