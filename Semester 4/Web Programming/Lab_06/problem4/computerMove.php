<?php

$data = json_decode(file_get_contents("php://input"), true);

$table = $data['table'];

function isWinning($table, $player) {
  for ($i = 0; $i < 3; $i++) {
    if ($table[$i][0] == $player && $table[$i][0] == $table[$i][1] && $table[$i][1] == $table[$i][2]) {
      return true;
    }
  }
  for ($i = 0; $i < 3; $i++) {
    if ($table[0][$i] == $player && $table[0][$i] == $table[1][$i] && $table[1][$i] == $table[2][$i]) {
      return true;
    }
  }
  if ($table[0][0] == $player && $table[0][0] == $table[1][1] && $table[1][1] == $table[2][2]) {
    return true;
  }
  if ($table[0][2] == $player && $table[0][2] == $table[1][1] && $table[1][1] == $table[2][0]) {
    return true;
  }
  return false;
}

$freeCells = 0;

foreach ($table as $row) {
    foreach ($row as $cell) {
        if ($cell == '') {
            $freeCells++;
        }
    }
}

if (isWinning($table, "x")) {
    echo "x";
    exit();
}

if ($freeCells == 0) {
    echo "r";
    exit();
}

$row = rand(0, 2);
$column = rand(0, 2);

while ($table[$row][$column] != '') {
    $row = rand(0, 2);
    $column = rand(0, 2);
}

$table[$row][$column] = "o";
echo $row . $column;

if (isWinning($table, "o")) {
    echo "o";
    exit();
}

if ($freeCells == 1) {
    echo "r";
    exit();
}

?>