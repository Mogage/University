<?php
session_start();

if (isset($_SESSION[session_id()])) {
    if ($_SESSION['rol'] == "admin")
        header("Location: admin.php");
    else
        header("Location: user.php");
    exit;
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $_POST['username'];
    $password = $_POST['password'];

    $connection = new PDO("sqlite:../database.db");

    $statement = $connection->prepare("SELECT * FROM users WHERE username = :username AND password = :password");
    $statement->bindParam(':username', $username);
    $statement->bindParam(':password', $password);
    $statement->execute();
    $profesors = $statement->fetchAll();
    if (sizeof($profesors) == 0) {
        $error = "Nume de utilizator sau parolă incorecte!";
    } else {
        session_id($profesors[0]['id']);
        $_SESSION[session_id()] = 1;
        if ($profesors[0]['id'] == 1) {
            $_SESSION['rol'] = "admin";
            header("Location: admin.php");
        } else {
            $_SESSION['rol'] = "user";
            header("Location: user.php");
        }
        exit;
    }
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>Autentificare utilizator</title>
</head>
<body>
    <h1>Autentificare utilizator</h1>

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