CREATE DATABASE SGBD_01;
USE SGBD_01;
GO

CREATE TABLE Joburi (
	cod_j INT PRIMARY KEY IDENTITY,
	nume VARCHAR(100),
	salariu REAL,
	nr_ore INT
);
GO

INSERT INTO Joburi (nume, salariu, nr_ore) VALUES
('Inginer programator', 99999, 6),
('Consultant financiar', 250, 12),
('Scrum Master', 10000000, 4),
('Tester', 8888, 24);

SELECT * FROM Joburi;