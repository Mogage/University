<?php
session_start();

if (isset($_SESSION['profesor_id'])) {
    header("Location: completare_note.php");
    exit;
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $_POST['username'];
    $password = $_POST['password'];

    $connection = new PDO("sqlite:../database.db");

    $statement = $connection->prepare("SELECT * FROM profesori WHERE username = :username AND password = :password");
    $statement->bindParam(':username', $username);
    $statement->bindParam(':password', $password);
    $statement->execute();
    $profesors = $statement->fetchAll();
    if (sizeof($profesors) == 0) {
        $error = "Nume de utilizator sau parolă incorecte!";
    }else {
        $_SESSION['profesor_id'] = 1;
        header("Location: completare_note.php");
        exit;
    }
}
?>

<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Autentificare profesor</title>
</head>
<body>
    <h1>Autentificare profesor</h1>

    <?php if (isset($error)) { ?>
        <p><?php echo $error; ?></p>
    <?php } ?>

    <form method="POST" action="<?php echo $_SERVER['PHP_SELF']; ?>">
        <label for="username">Nume de utilizator:</label>
        <input type="text" name="username" id="username" required>
        <br>
        <label for="password">Parolă:</label>
        <input type="password" name="password" id="password" required>
        <br>
        <input type="submit" value="Autentificare">
    </form>
</body>
</html>