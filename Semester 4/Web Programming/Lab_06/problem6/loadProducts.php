<?php

$connection = new PDO("sqlite:../database.db");

$sql = "SELECT * FROM prob6";
$products = array();

if (isset($_GET['producator'])) {
    $producator = $_GET['producator'];
    foreach ($producator as $producator) {
        $sql = "SELECT * FROM prob6 WHERE producator = :producator";
        $stmt = $connection->prepare($sql);
        $stmt->bindParam(':producator', $producator);
        $stmt->execute();
        $products = array_merge($products, $stmt->fetchAll());
    }

}
if (isset($_GET['procesor'])) {
    $procesor = $_GET['procesor'];
    foreach ($procesor as $procesor) {
        $sql = "SELECT * FROM prob6 WHERE procesor = :procesor";
        $stmt = $connection->prepare($sql);
        $stmt->bindParam(':procesor', $procesor);
        $stmt->execute();
        $products = array_merge($products, $stmt->fetchAll());
    }
}
if (isset($_GET['memorie'])) {
    $memorie = $_GET['memorie'];
    foreach ($memorie as $memorie) {
        $sql = "SELECT * FROM prob6 WHERE memorie = :memorie";
        $stmt = $connection->prepare($sql);
        $stmt->bindParam(':memorie', $memorie);
        $stmt->execute();
        $products = array_merge($products, $stmt->fetchAll());
    }
}
if (isset($_GET['capacitate'])) {
    $capacitate = $_GET['capacitate'];
    foreach ($capacitate as $capacitate) {
        $sql = "SELECT * FROM prob6 WHERE capacitate = :capacitate";
        $stmt = $connection->prepare($sql);
        $stmt->bindParam(':capacitate', $capacitate);
        $stmt->execute();
        $products = array_merge($products, $stmt->fetchAll());
    }
}
if (isset($_GET['an'])) {
    $an = $_GET['an'];
    foreach ($an as $an) {
        $sql = "SELECT * FROM prob6 WHERE an = :an";
        $stmt = $connection->prepare($sql);
        $stmt->bindParam(':an', $an);
        $stmt->execute();
        $products = array_merge($products, $stmt->fetchAll());
    }
}
if (isset($_GET['pret'])) {
    $pret = $_GET['pret'];
    foreach ($pret as $pret) {
        $sql = "SELECT * FROM prob6 WHERE pret = :pret";
        $stmt = $connection->prepare($sql);
        $stmt->bindParam(':pret', $pret);
        $stmt->execute();
        $products = array_merge($products, $stmt->fetchAll());
    }
}

if (empty($_GET)) {
    $stmt = $connection->prepare($sql);
    $stmt->execute();
    $products = array_merge($products, $stmt->fetchAll());
}

echo "<tr>";
echo "<th>Producator</th>";
echo "<th>Procesor</th>";
echo "<th>Memorie RAM</th>";
echo "<th>Capacitate</th>";
echo "<th>An</th>";
echo "<th>Pret</th>";
echo "</tr>";

foreach ($products as $product) {
    echo "<tr>";
    echo "<td>" . $product['producator'] . "</td>";
    echo "<td>" . $product['procesor'] . "</td>";
    echo "<td>" . $product['memorie'] . "</td>";
    echo "<td>" . $product['capacitate'] . "</td>";
    echo "<td>" . $product['an'] . "</td>";
    echo "<td>" . $product['pret'] . "</td>";
    echo "</tr>";
}

?>