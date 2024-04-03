USE Laboratory_1;

DROP TABLE OrganizationsCountries;
DROP TABLE Countries;
DROP TABLE Organizations;


CREATE TABLE Organizations (
	organization_code INT PRIMARY KEY,
	orgnaization_name VARCHAR(50) NOT NULL,
	headquarter VARCHAR(50) NOT NULL,
	founded_date DATE NOT NULL
);

CREATE TABLE Countries (
	country_code INT PRIMARY KEY,
	country_name VARCHAR(50) NOT NULL,
	capital VARCHAR(50) NOT NULL,
	area VARCHAR(50) NOT NULL,
	population INT NOT NULL,
	continent VARCHAR(25) NOT NULL,
);

CREATE TABLE OrganizationsCountries (
	code INT PRIMARY KEY,
	organization_code INT FOREIGN KEY REFERENCES Organizations(organization_code),
	country_code INT FOREIGN KEY REFERENCES Countries(country_code)
);

INSERT INTO Organizations (organization_code, orgnaization_name, headquarter, founded_date)
VALUES
    (1, N'Tech Innovators Inc.', N'Silicon Valley', '1990-06-15'),
    (2, N'Global Solutions Co.', N'New York City', '2005-03-20'),
    (3, N'Healthcare Foundation', N'London', '1985-12-10'),
    (4, N'NATO', N'Brussels', '1949-04-04'); 

INSERT INTO Countries (country_name, country_code, capital, area, population, continent)
VALUES
    (N'United States', 1, N'Washington D.C.', 9834000, 328200000, N'North America'),
    (N'United Kingdom', 2, N'London', 242495, 66436000, N'Europe'),
    (N'India', 3, N'New Delhi', 3287240, 1380004385, N'Asia'),
    (N'Belgium', 4, N'Brussels', 30528, 11433256, N'Europe');

INSERT INTO OrganizationsCountries(code, country_code, organization_code)
VALUES
    (1, 1, 1), 
    (2, 2, 1), 
    (3, 3, 1), 
    (4, 1, 2), 
    (5, 2, 3), 
    (6, 4, 4); 

SELECT * 
FROM Countries as c
INNER JOIN OrganizationsCountries as co ON co.country_code = c.country_code
INNER JOIN Organizations as o ON o.organization_code = CO.organization_code
WHERE o.orgnaization_name = 'NATO';

SELECT * 
FROM Countries as c
INNER JOIN OrganizationsCountries as co ON co.country_code = c.country_code
INNER JOIN Organizations as o ON o.organization_code = CO.organization_code
WHERE o.founded_date < '1980-01-01';

SELECT c.country_code, c.country_name
FROM OrganizationsCountries as co
INNER JOIN Countries as c ON co.country_code = c.country_code
GROUP BY c.country_code, c.country_name
HAVING COUNT(co.country_code) = 1;

SELECT c.capital
FROM Countries c
LEFT JOIN Organizations o ON c.capital = o.headquarter
WHERE o.organization_code IS NULL;

SELECT continent, SUM(population) AS total_population
FROM Countries
GROUP BY continent;

SELECT continent, COUNT(*) AS total_countries
FROM Countries
GROUP BY continent;
