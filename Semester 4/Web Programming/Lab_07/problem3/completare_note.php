<?php
session_start();

if (!isset($_SESSION['profesor_id'])) {
    header("Location: index.php");
    exit;
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $student_id = $_POST['student_id'];
    $materie = $_POST['materie'];
    $nota = $_POST['nota'];


    $connection = new PDO("sqlite:../database.db");

    $statementStudents = $connection->prepare("SELECT * FROM studenti WHERE id = :student_id");
    $statementStudents->bindParam(':student_id', $student_id);
    $statementStudents->execute();

    $statementSubjects = $connection->prepare("SELECT * FROM materii WHERE LOWER(nume) = LOWER(:materie)");
    $statementSubjects->bindParam(':materie', $materie);
    $statementSubjects->execute();

    if ($nota < 1 || $nota > 10) {
        $error = "Nota trebuie să fie între 1 și 10";
    }
    else if (sizeof($statementStudents->fetchAll()) == 0) {
        $error = "There is no student with this id";
    }
    else if (sizeof($statementSubjects->fetchAll()) == 0){
        $error = "There is no subject with this name";
    }
    else {
        $statement = $connection->prepare("INSERT INTO note (id_student, nume_materie, nota) VALUES (:student_id, LOWER(:materie), :nota)");
        $statement->bindParam(':student_id', $student_id);
        $statement->bindParam(':materie', $materie);
        $statement->bindParam(':nota', $nota);

        if (!$statement->execute()) {
            $error = "There was an error saving the grade";
        }
        $success = "Nota a fost salvată cu succes!";
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
    <title>Completare note</title>
</head>
<body>
    <h1>Completare note</h1>

    <?php if (isset($success)) { ?>
        <p><?php echo $success; ?></p>
    <?php } ?>

    <?php if (isset($error)) { ?>
            <p><?php echo $error; ?></p>
        <?php } ?>

    <form method="POST" action="<?php echo $_SERVER['PHP_SELF']; ?>">
        <label for="student_id">ID student:</label>
        <input type="text" name="student_id" id="student_id" required>
        <br>
        <label for="materie">Materie:</label>
        <input type="text" name="materie" id="materie" required>
        <br>
        <label for="nota">Nota:</label>
        <input type="text" name="nota" id="nota" required>
        <br>
        <input type="submit" value="Salvează">
    </form>

    <form method="GET" action="<?php echo $_SERVER['PHP_SELF']; ?>">
            <input type="hidden" name="logout" value="1">
            <input type="submit" value="Log Out">
        </form>
</body>
</html>
