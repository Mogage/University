<?php
session_start();

if (!isset($_SESSION[session_id()])) {
    header("Location: index.php");
    exit;
}

if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_FILES["image"])) {
    $targetDir = "imagini/";
    $targetFile = $targetDir . basename($_FILES["image"]["name"]);

    $imageFileType = strtolower(pathinfo($targetFile, PATHINFO_EXTENSION));
    if ($imageFileType === "jpg" || $imageFileType === "jpeg" || $imageFileType === "png") {
        if (move_uploaded_file($_FILES["image"]["tmp_name"], $targetFile)) {
            $success = "Imaginea a fost încărcată cu succes!";
        } else {
            $error = "A apărut o eroare la încărcarea imaginii!";
        }
    } else {
        $error = "Formatul fișierului nu este suportat. Sunt acceptate doar fișiere de tip JPG, JPEG și PNG!";
    }
}

// Verificați dacă s-a trimis cererea de ștergere a imaginii
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST["delete_image"])) {
    $imageToDelete = $_POST["delete_image"];
        $imageDirectory = "imagini/";

    if (file_exists($imageToDelete)) {
        if (unlink($imageToDelete)) {
            $success = "Imaginea a fost ștearsă cu succes!";
        } else {
            $error = "A apărut o eroare la ștergerea imaginii!";
        }
    } else {
        $error = "Imaginea nu există!";
    }
}

if (isset($_GET['logout'])) {
    session_destroy(); // Distruge sesiunea
    header("Location: index.php");
    exit;
}

?>

<!DOCTYPE html>
<html>
<head>
    <title>Profil utilizator</title>
</head>
<body>
    <h1>Profil utilizator</h1>

    <?php if (isset($success)) { ?>
        <p><?php echo $success; ?></p>
    <?php } ?>

    <h2>Încărcați o imagine</h2>

    <?php if (isset($error)) { ?>
        <p><?php echo $error; ?></p>
    <?php } ?>

    <form method="POST" action="<?php echo $_SERVER['PHP_SELF']; ?>" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <br>
        <input type="submit" value="Încărcați imaginea">
    </form>

    <h2>Imaginile încărcate:</h2>

    <?php
    $imageDirectory = "imagini/";
    $images = glob($imageDirectory . "*.{jpg,jpeg,png}", GLOB_BRACE);

    if ($images) {
        foreach ($images as $image) {
            echo '<img src="' . $image . '" width="200" height="200">';
            echo '<form method="POST" action="' . $_SERVER['PHP_SELF'] . '">';
            echo '<input type="hidden" name="delete_image" value="' . $image . '">';
            echo '<input type="submit" value="Șterge">';
            echo '</form>';
        }
    } else {
        echo '<p>Nu există imagini încărcate.</p>';
    }
    ?>

    <form method="GET" action="<?php echo $_SERVER['PHP_SELF']; ?>">
        <input type="hidden" name="logout" value="1">
        <input type="submit" value="Log Out">
    </form>
</body>
</html>
