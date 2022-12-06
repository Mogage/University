USE Laborator_01;
GO;

CREATE VIEW ViewUsersProfiles AS
SELECT U.userEmail, P.profileName
FROM Users AS U 
INNER JOIN Profiles AS P ON U.idUser = P.idUser;

SELECT * FROM ViewUsersProfiles;

CREATE VIEW ViewRatings AS 
SELECT P.profileName, P2.productionTitle, R.ratingValue
FROM Ratings AS R
INNER JOIN Profiles AS P ON P.idProfile = R.idProfile
INNER JOIN Productions AS P2 ON R.idProduction = P2.idProduction;

SELECT * FROM ViewRatings;

CREATE VIEW ViewProductions AS
SELECT productionTitle, productionType
FROM Productions;

SELECT * FROM ViewProductions;

CREATE VIEW ViewProducers AS
SELECT producerFirstName, producerLastName
FROM Producers;

SELECT * FROM ViewProducers;

CREATE VIEW ViewPPP AS
SELECT P1.productionTitle, P2.producerFirstName, P2.producerLastName, PP.salary
FROM ProductionsProducers AS PP
INNER JOIN Productions AS P1 ON PP.idProduction = P1.idProduction
INNER JOIN Producers AS P2 ON PP.idProducer = P2.idProducer;

SELECT * FROM ViewPPP;