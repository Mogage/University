USE Laborator_01;

-- 1. selecteaza toti utilizatorii si profilele corespunzatoare
SELECT [Email] = U.userEmail, P.profileName as [Profile]
FROM Profiles P 
	INNER JOIN Users U ON p.idUser = U.idUser
GROUP BY U.userEmail, P.profileName
ORDER BY u.userEmail, P.profileName;

-- 2. selecteaza productiile de drama care le corespund
SELECT P.productionTitle as [Title], G.genreName as [Genre]
FROM  ProductionsGenres PG 
	INNER JOIN Productions P ON P.idProduction = PG.idProduction
	INNER JOIN Genres G ON PG.idGenre = G.idGenre
WHERE G.genreName = 'Drama'
GROUP BY G.genreName, P.productionTitle;

-- 3. selecteaza toate recenziile diferite a unei productii
SELECT DISTINCT P.productionTitle as [Title],R.ratingValue as [Rating]
FROM Productions P
	INNER JOIN Ratings R ON p.idProduction = R.idProduction;

-- 4. selecteaza productiile cu rating peste 5
SELECT P.productionTitle as [Title], AVG(R.ratingValue) as [Average rating]
FROM Productions P 
	INNER JOIN Ratings R ON P.idProduction=R.idProduction
WHERE P.productionType = 'serial'
GROUP BY P.idProduction, P.productionTitle
HAVING AVG(R.ratingValue) > 5;

-- 5. selecteaza productiile si numele regizorilor cu salariul primit
SELECT Productions.productionTitle as [Title], Producers.producerFirstName + ' ' + Producers.producerLastName as [Producer Name], ProductionsProducers.salary as [Salary]
FROM Productions, Producers, ProductionsProducers
WHERE ProductionsProducers.idProducer = Producers.idProducer AND Productions.idProduction = ProductionsProducers.idProduction
GROUP BY Productions.productionTitle, Producers.producerFirstName,  Producers.producerLastName, ProductionsProducers.salary;

-- 6. selecteaza productiile si numele actorilor
SELECT P.productionTitle as [Title], A.actorFirstName + ' ' + A.actorLastName as [Actors], PA.characterName as [Character]
FROM ProductionsActors PA
	INNER JOIN Actors A ON PA.idActor = A.idActor
	INNER JOIN Productions P ON PA.idProduction = P.idProduction
GROUP BY P.productionTitle, A.actorFirstName, A.actorLastName, PA.characterName;

-- 7. selecteaza filmele si arata durata lor
SELECT P.productionTitle as [Title], E.episodeMinutesLength as [Minutes Length]
FROM Productions P 
	INNER JOIN Episodes E ON  P.idProduction = E.idProduction
WHERE P.productionType = 'movie';

-- 8. selecteaza episoadele serialelor cu durata mai mica de 50 de minute
SELECT P.productionTitle as [Title], E.episodeName as [Episode Name], E.episodeMinutesLength as [Minutes Length]
FROM Productions P 
	INNER JOIN Episodes E ON P.idProduction = E.idProduction
WHERE P.productionType = 'serial'
GROUP BY P.productionTitle, E.episodeName, E.episodeMinutesLength
HAVING E.episodeMinutesLength < 50;

-- 9. selecteaza productiile si numarul de sezoane pe care le are fieacare
SELECT P.productionTitle as [Name], COUNT(DISTINCT E.episodeSeason) as [Number of seasons]
FROM Episodes E 
	INNER JOIN Productions P ON P.idProduction = E.idProduction
GROUP BY P.productionTitle;

-- 10 selecteaza 
SELECT P.profileName as [Name], AVG(R.ratingValue) as [Average Rating]
FROM Profiles P 
	INNER JOIN Ratings R ON P.idProfile = R.idProfile
GROUP BY P.profileName, R.ratingValue;
