<?php

// echo "<div";
// foreach ($destinations as $destination) {
//     echo "<p style=\"font-size: 20px\">" . $destination['plecare'] . " -- > " .  $destination['destinatie'] . "</p>";
// }
// echo "</div>";

function findAllRoutes($graph, $start, $end, $visited, $route) {
    $visited[$start] = true;
    $route[] = $start;

    if ($start == $end) {
        echo "<p style=\"font-size: 30px\">" . implode(' -> ', array_map('htmlspecialchars', $route)) . "</p>";
    } else {
        foreach ($graph[$start] as $nextStation) {
            if (!$visited[$nextStation]) {
                findAllRoutes($graph, $nextStation, $end, $visited, $route);
            }
        }
    }

    $lastStationIndex = array_search($start, $route);
    unset($visited[$start]);
    array_splice($route, $lastStationIndex);
}

function withChanges($connection) {
    $sql = "SELECT * FROM trenuri";

    $stmt = $connection->prepare($sql);
    $stmt->execute();

    $trainConnections = $stmt->fetchAll();

    $trainConnectionsGraph = [];
    $visited = [];
    foreach ($trainConnections as $trainConnection) {
        $trainConnectionsGraph[$trainConnection['plecare']][] = $trainConnection['destinatie'];
        $visited[$trainConnection['plecare']] = false;
        $visited[$trainConnection['destinatie']] = false;
    }

    $route = [];
    findAllRoutes($trainConnectionsGraph, htmlspecialchars($_POST['plecare'], ENT_QUOTES, 'UTF-8'),
                htmlspecialchars($_POST['destinatie'], ENT_QUOTES, 'UTF-8'), $visited, $route);
}

function withoutChanges($connection) {
    $sql = "SELECT * FROM trenuri WHERE plecare LIKE :plecare AND destinatie LIKE :destinatie";

    $stmt = $connection->prepare($sql);
    $stmt->execute(['plecare' => htmlspecialchars($_POST['plecare'], ENT_QUOTES, 'UTF-8') . '%',
                    'destinatie' => htmlspecialchars($_POST['destinatie'], ENT_QUOTES, 'UTF-8') . '%']);

    $destinations = $stmt->fetchAll();

    foreach ($destinations as $destination) {
        echo "<p style=\"font-size: 30px\">" . htmlspecialchars($destination['plecare'], ENT_QUOTES, 'UTF-8') . " -- > "
         .  htmlspecialchars($destination['destinatie'], ENT_QUOTES, 'UTF-8') . "</p>";
    }
}

$connection = new PDO("sqlite:../database.db");

$checkboxValue = isset($_POST['legatura']) ? htmlspecialchars($_POST['legatura'], ENT_QUOTES, 'UTF-8') : "false";
if ($checkboxValue == "true") {
    withChanges($connection);
} else {
    withoutChanges($connection);
}

?>