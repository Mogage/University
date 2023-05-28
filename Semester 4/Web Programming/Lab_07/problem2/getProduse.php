<?php

$connection = new PDO("sqlite:../database.db");

$perPage = $_POST["per_page"] ?? 3;

$totalCountQuery = "SELECT COUNT(*) AS total FROM produse";
$statement = $connection->prepare($totalCountQuery);
$statement->execute();
$totalCount = $statement->fetch(PDO::FETCH_ASSOC)["total"];

$totalPages = ceil($totalCount / $perPage);

$currentPage = $_GET["page"] ?? 1;
$currentPage = max(1, min($currentPage, $totalPages));

$offset = ($currentPage - 1) * $perPage;
$limit = $perPage;

$query = "SELECT * FROM produse LIMIT $offset, $limit";
$statement = $connection->prepare($query);
$statement->execute();
$result = $statement->fetchAll();

if (sizeof($result) > 0) {
    foreach ($result as $row) {
        echo "<p>" . $row["nume"] . " - " . $row["pret"] . " - " . $row["expirare"] . "</p>";
    }
} else {
    echo "Nu sunt produse disponibile.";
}

if ($totalPages > 1) {
    echo "<p> Pagina $currentPage din $totalPages </p>";
    echo "<p>";
    if ($currentPage > 1) {
        $previousPage = $currentPage - 1;
        echo "<a href='getProduse.php?page=$previousPage'>Previous</a> ";
    }

    if ($currentPage < $totalPages) {
        $nextPage = $currentPage + 1;
        echo "<a href='getProduse.php?page=$nextPage'>Next</a>";
    }

    echo "</p>";
}

?>