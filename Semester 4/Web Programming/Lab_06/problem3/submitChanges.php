<?php

$connection = new PDO("sqlite:../database.db");

$sql = "UPDATE prob3 SET nume = :nume, prenume = :prenume, email = :email, oras = :oras, strada = :strada WHERE id = :id";

$data = json_decode(file_get_contents("php://input"), true);

$stmt = $connection->prepare($sql);
$stmt->bindValue(':id', $data['id']);
$stmt->bindValue(':nume', $data['nume']);
$stmt->bindValue(':prenume', $data['prenume']);
$stmt->bindValue(':email', $data['email']);
$stmt->bindValue(':oras', $data['oras']);
$stmt->bindValue(':strada', $data['strada']);
// $stmt->bindValue(':id', 1);
// $stmt->bindValue(':nume', "nume1");
// $stmt->bindValue(':prenume', "prenume1");
// $stmt->bindValue(':email', "email1@yahoo.com");
// $stmt->bindValue(':oras', "oras1");
// $stmt->bindValue(':strada', "strada6");
$result = $stmt->execute();

if ($result) {
    echo "Success";
}
else {
    echo "Fail";
}

?>