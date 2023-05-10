--Titles(id-pk int, genres-fk, name varchar, type varchar)
--Ratings(id-pk int, value varchar)
--Episodes(id-pk int, titles_id-fk, name vachar, durata_minute int)
--Users(id-pk int, email varchar, password varchar)
--Profiles(id-pk int, users_id-fk, name varchar, photo image)
--Playlists|My List(id-pk int, profiles_id-fk, titles_id-fk)
--Genres(id-pk int, name varchar)
--Actors(id-pk, first_name varchar, last_name varchar)
--Producers(id-pk, first_name varchar, last_name varchar)
--TitlesActorsAuxiliary(titles_id-fk, actors_id-fk)
--TitlesProducersAuxiliary(titles_id-fk, producers_id-fk)
--CREATE TABLE MyLists(idProfile INT FOREIGN KEY REFERENCES Profiles, idProduction INT FOREIGN KEY REFERENCES Productions);

--CREATE DATABASE Laborator_01;

USE Laborator_01;

CREATE TABLE Productions(
	idProduction INT PRIMARY KEY IDENTITY,
	productionTitle VARCHAR(50) NOT NULL,
	productionType VARCHAR(50) NOT NULL
);

CREATE TABLE Episodes(
	idEpisode INT PRIMARY KEY IDENTITY,
	idProduction INT FOREIGN KEY REFERENCES Productions(idProduction),
	episodeName VARCHAR(50) NOT NULL,
	episodeMinutesLength INT NOT NULL
);

CREATE TABLE Users(
	idUser INT PRIMARY KEY IDENTITY,
	userEmail VARCHAR(50) NOT NULL,
	userPassword BINARY(50) NOT NULL
);

CREATE TABLE Profiles(
	idProfile INT PRIMARY KEY IDENTITY,
	idUser INT FOREIGN KEY REFERENCES Users(idUser),
	profileName VARCHAR(50) NOT NULL
);

CREATE TABLE Ratings(
	idProfile INT,
	idProduction INT,
	ratingValue INT CHECK (ratingValue >= 1 AND ratingValue <= 10),
	CONSTRAINT fkProfileRating FOREIGN KEY (idProfile) REFERENCES Profiles(idProfile),
	CONSTRAINT fkProductionRating FOREIGN KEY (idProduction) REFERENCES Productions(idProduction),
	CONSTRAINT pkRating PRIMARY KEY (idProfile, idProduction)
);

CREATE TABLE Genres(
	idGenre INT PRIMARY KEY IDENTITY,
	genreName VARCHAR(50) NOT NULL
);

CREATE TABLE ProductionsGenres(
	idProduction INT,
	idGenre INT,
	CONSTRAINT fkProductionGenre FOREIGN KEY (idProduction) REFERENCES Productions(idProduction),
	CONSTRAINT fkGenre FOREIGN KEY (idGenre) REFERENCES GENRES(idGenre),
	CONSTRAINT pkProductiongsGenres PRIMARY KEY (idProduction, idGenre)
);

CREATE TABLE Actors(
	idActor INT PRIMARY KEY IDENTITY,
	actorFirstName VARCHAR(50) NOT NULL,
	actorLastName VARCHAR(50) NOT NULL
);

CREATE TABLE Producers(
	idProducer INT PRIMARY KEY IDENTITY,
	producerFirstName VARCHAR(50) NOT NULL,
	producerLastName VARCHAR(50) NOT NULL
);

CREATE TABLE ProductionsActors(
	idProduction INT,
	idActor INT,
	CONSTRAINT fkProductionActor FOREIGN KEY (idProduction) REFERENCES Productions(idProduction),
	CONSTRAINT fkActor FOREIGN KEY (idActor) REFERENCES Actors(idActor),
	CONSTRAINT pkProductionsActors PRIMARY KEY (idProduction, idActor)
);

CREATE TABLE ProductionsProducers(
	idProduction INT,
	idProducer INT,
	salary INT,
	CONSTRAINT fkProductionProducer FOREIGN KEY (idProduction) REFERENCES Productions(idProduction),
	CONSTRAINT fkProducer FOREIGN KEY (idProducer) REFERENCES Producers(idProducer),
	CONSTRAINT pkProductionsProducers PRIMARY KEY (idProduction, idProducer)
);
GO

INSERT INTO Producers (producerFirstName, producerLastName) VALUES ('Ryan', 'Murphy');
INSERT INTO Producers (producerFirstName, producerLastName) VALUES ('Ian', 'Brennan');
INSERT INTO Producers (producerFirstName, producerLastName) VALUES ('Grant', 'Harvey');
INSERT INTO Producers (producerFirstName, producerLastName) VALUES ('Vince', 'Gilligan');
INSERT INTO Producers (producerFirstName, producerLastName) VALUES ('Michael', 'Slovis');
INSERT INTO Producers (producerFirstName, producerLastName) VALUES ('John', 'Hamburg');

INSERT INTO Productions (productionTitle, productionType) VALUES ('Povestea lui Jeffrey Dahmer', 'serial');
INSERT INTO Productions (productionTitle, productionType) VALUES ('Breaking Bad', 'serial');
INSERT INTO Productions (productionTitle, productionType) VALUES ('Me time', 'movie');

INSERT INTO Actors (actorFirstName, actorLastName) VALUES ('Kevin', 'Hart');
INSERT INTO Actors (actorFirstName, actorLastName) VALUES ('Mark', 'Wahlberg');
INSERT INTO Actors (actorFirstName, actorLastName) VALUES ('Evan', 'Peters');
INSERT INTO Actors (actorFirstName, actorLastName) VALUES ('Richiard', 'Jenkins');
INSERT INTO Actors (actorFirstName, actorLastName) VALUES ('Bryan', 'Cranston');
INSERT INTO Actors (actorFirstName, actorLastName) VALUES ('Aaron', 'Paul');

INSERT INTO Genres (genreName) VALUES ('Drama');
INSERT INTO Genres (genreName) VALUES ('Thriller');
INSERT INTO Genres (genreName) VALUES ('Crime series');
INSERT INTO Genres (genreName) VALUES ('Comedy');

INSERT INTO Users (userEmail, userPassword) VALUES ('denis.moldovan@gmail.com', 'Denis123');
INSERT INTO Users (userEmail, userPassword) VALUES ('daria.marian@gmail.com', 'Daria123');

INSERT INTO Profiles (idUser, profileName) VALUES (1, 'denis');
INSERT INTO Profiles (idUser, profileName) VALUES (1, 'antonela');
INSERT INTO Profiles (idUser, profileName) VALUES (1, 'nicu');
INSERT INTO Profiles (idUser, profileName) VALUES (2, 'daria');
INSERT INTO Profiles (idUser, profileName) VALUES (2, 'eveline');
INSERT INTO Profiles (idUser, profileName) VALUES (2, 'dragos');

INSERT INTO Ratings (idProfile, idProduction, ratingValue) VALUES (3, 2, 10);
INSERT INTO Ratings (idProfile, idProduction, ratingValue) VALUES (1, 1, 9);
INSERT INTO Ratings (idProfile, idProduction, ratingValue) VALUES (4, 3, 4);
INSERT INTO Ratings (idProfile, idProduction, ratingValue) VALUES (5, 2, 6);
INSERT INTO Ratings (idProfile, idProduction, ratingValue) VALUES (6, 1, 10);

INSERT INTO Episodes (idProduction, episodeName, episodeMinutesLength) VALUES (1, 'Episode one', 49);
INSERT INTO Episodes (idProduction, episodeName, episodeMinutesLength) VALUES (2, 'Alegerea', 48);
INSERT INTO Episodes (idProduction, episodeName, episodeMinutesLength) VALUES (2, 'Derapaj', 48);
INSERT INTO Episodes (idProduction, episodeName, episodeMinutesLength) VALUES (1, 'Please, dont leave', 52);
INSERT INTO Episodes (idProduction, episodeName, episodeMinutesLength) VALUES (3, 'Me Time', 104);
INSERT INTO Episodes (idProduction, episodeName, episodeMinutesLength) VALUES (2, 'Pilot', 58);
INSERT INTO Episodes (idProduction, episodeName, episodeMinutesLength) VALUES (1, 'Do as Dahmer',53);

INSERT INTO ProductionsActors (idProduction, idActor) VALUES (1, 3);
INSERT INTO ProductionsActors (idProduction, idActor) VALUES (1, 4);
INSERT INTO ProductionsActors (idProduction, idActor) VALUES (2, 5);
INSERT INTO ProductionsActors (idProduction, idActor) VALUES (2, 6);
INSERT INTO ProductionsActors (idProduction, idActor) VALUES (3, 1);
INSERT INTO ProductionsActors (idProduction, idActor) VALUES (3, 2);

INSERT INTO ProductionsProducers (idProduction, idProducer, salary) VALUES (1, 1, 20);
INSERT INTO ProductionsProducers (idProduction, idProducer, salary) VALUES (1, 2, 200);
INSERT INTO ProductionsProducers (idProduction, idProducer, salary) VALUES (1, 3, 500);
INSERT INTO ProductionsProducers (idProduction, idProducer, salary) VALUES (2, 4, 1);
INSERT INTO ProductionsProducers (idProduction, idProducer, salary) VALUES (2, 5, 1000);
INSERT INTO ProductionsProducers (idProduction, idProducer, salary) VALUES (3, 6, 243);

INSERT INTO ProductionsGenres (idProduction, idGenre) VALUES (1, 1);
INSERT INTO ProductionsGenres (idProduction, idGenre) VALUES (1, 3);
INSERT INTO ProductionsGenres (idProduction, idGenre) VALUES (2, 1);
INSERT INTO ProductionsGenres (idProduction, idGenre) VALUES (2, 2);
INSERT INTO ProductionsGenres (idProduction, idGenre) VALUES (2, 3);
INSERT INTO ProductionsGenres (idProduction, idGenre) VALUES (3, 4);

--DROP TABLE ProductionsGenres;
--DROP TABLE ProductionsActors;
--DROP TABLE ProductionsProducers;
--DROP TABLE Episodes;
--DROP TABLE Ratings;
--DROP TABLE Profiles;
--DROP TABLE Productions;
--DROP TABLE Users;
--DROP TABLE Genres;
--DROP TABLE Actors;
--DROP TABLE Producers;