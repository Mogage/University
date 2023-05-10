<?php

$name = $_GET["q"];

if (is_file($name)) {
    echo "<div onclick=loadDirectory('../..')>..</div>";
    echo strip_tags(file_get_contents($name), "<html>");
    exit();
}

$files = scandir($name);
foreach($files as $file) {
    echo "<li><div onclick=loadDirectory('" . $_GET["q"] . "/" . $file . "')>" . $file . "</div></li>";
}

?>