<?php
session_start();

// Conectarea la baza de date
$connection = new PDO("sqlite:../database.db");

// Verificăm dacă s-a trimis un comentariu
if ($_SERVER["REQUEST_METHOD"] === "POST" && isset($_POST["comentariu"])) {
    $nume = $_POST["nume"];
    $comentariu = $_POST["comentariu"];

    // Salvăm comentariul în baza de date cu starea "neaprobat"
    $query = $connection->prepare("INSERT INTO comentarii (nume, comentariu, aprobat) VALUES (?, ?, 0)");
    $query->execute([$nume, $comentariu]);
}

// Obținem toate comentariile aprobate din baza de date
$query = $connection->prepare("SELECT * FROM comentarii WHERE aprobat = 1");
$query->execute();
$comentarii = $query->fetchAll(PDO::FETCH_ASSOC);

if (isset($_GET['logout'])) {
    session_destroy(); // Distruge sesiunea
    header("Location: index.php");
    exit;
}

?>

<!DOCTYPE html>
<html>
<head>
    <title>Articol</title>
</head>
<body>
    <h1>Utilizator</h1>

    <!-- Afisam comentariile aprobate -->
    <?php if ($comentarii): ?>
        <h2>Comentarii</h2>
        <ul>
            <?php foreach ($comentarii as $comentariu): ?>
                <li>
                    <strong><?php echo htmlspecialchars($comentariu["nume"]); ?>:</strong>
                    <?php echo htmlspecialchars($comentariu["comentariu"]); ?>
                </li>
            <?php endforeach; ?>
        </ul>
    <?php else: ?>
        <p>Nu există comentarii aprobate momentan.</p>
    <?php endif; ?>

    <!-- Formular pentru adăugarea unui comentariu -->
    <h2>Adaugă un comentariu</h2>
    <form method="POST" action="">
        <label for="nume">Nume:</label>
        <input type="text" name="nume" id="nume" required>

        <br>

        <label for="comentariu">Comentariu:</label>
        <textarea name="comentariu" id="comentariu" rows="4" required></textarea>

        <br>

        <button type="submit">Postează Comentariu</button>
    </form>

    <form method="GET" action="<?php echo $_SERVER['PHP_SELF']; ?>">
        <input type="hidden" name="logout" value="1">
        <input type="submit" value="Log Out">
    </form>
</body>
</html>
