<?php
session_start();

// Conectarea la baza de date
$connection = new PDO("sqlite:../database.db");

// Funcție pentru aprobarea unui comentariu
function aprobaComentariu($idComentariu) {
    global $connection;

    // Actualizăm starea comentariului în baza de date
    $query = $connection->prepare("UPDATE comentarii SET aprobat = 1 WHERE id_comentariu = ?");
    $query->execute([$idComentariu]);
}

if ($_SERVER["REQUEST_METHOD"] === "POST" && isset($_POST["id_comentariu_update"])) {
    aprobaComentariu($_POST["id_comentariu_update"]);
}

// Funcție pentru ștergerea unui comentariu
function stergeComentariu($idComentariu) {
    global $connection;

    // Ștergem comentariul din baza de date
    $query = $connection->prepare("DELETE FROM comentarii WHERE id_comentariu = ?");
    $query->execute([$idComentariu]);
}

if ($_SERVER["REQUEST_METHOD"] === "POST" && isset($_POST["id_comentariu_delete"])) {
    stergeComentariu($_POST["id_comentariu_delete"]);
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
    <title>Articol</title>
</head>
<body>
    <h1>Admin</h1>

    <form method="GET" action="<?php echo $_SERVER['PHP_SELF']; ?>">
        <input type="hidden" name="logout" value="1">
        <input type="submit" value="Log Out">
    </form>

    <!-- Partea de administrator -->
    <?php if ($_SESSION["rol"] === "admin"): ?>
        <h2>Moderare Comentarii</h2>
        <?php
        // Obținem toate comentariile neaprobate din baza de date
        $query = $connection->prepare("SELECT * FROM comentarii WHERE aprobat = 0");
        $query->execute();
        $comentariiNeaprobate = $query->fetchAll(PDO::FETCH_ASSOC);
        ?>

        <?php if ($comentariiNeaprobate): ?>
            <ul>
                <?php foreach ($comentariiNeaprobate as $comentariu): ?>
                    <li>
                        <strong><?php echo htmlspecialchars($comentariu["nume"]); ?>:</strong>
                        <?php echo htmlspecialchars($comentariu["comentariu"]); ?>

                        <!-- Buton pentru aprobarea comentariului -->
                        <form method="POST" action="">
                            <input type="hidden" name="id_comentariu_update" value="<?php echo $comentariu['id_comentariu']; ?>">
                            <button type="submit" name="aproba_comentariu">Aprobă Comentariu</button>
                        </form>

                        <!-- Buton pentru ștergerea comentariului -->
                        <form method="POST" action="">
                            <input type="hidden" name="id_comentariu_delete" value="<?php echo $comentariu['id_comentariu']; ?>">
                            <button type="submit" name="sterge_comentariu">Șterge Comentariu</button>
                        </form>
                    </li>
                <?php endforeach; ?>
            </ul>
        <?php else: ?>
            <p>Nu există comentarii de moderat momentan.</p>
        <?php endif; ?>
    <?php endif; ?>
</body>
</html>
