USE Laborator_01
GO

--CREATE TABLE DatabaseVersion(currentVersion SMALLINT);
--INSERT INTO DatabaseVersion(currentVersion) VALUES (0);
--GO

CREATE OR ALTER PROCEDURE ProcedureUp1
AS
BEGIN
	ALTER TABLE ProductionsProducers
	ALTER COLUMN salary SMALLINT
END
GO

CREATE OR ALTER PROCEDURE ProcedureUp2
AS
BEGIN
	ALTER TABLE Profiles
	ADD CONSTRAINT df_profileName DEFAULT 'user' FOR profileName
END
GO

CREATE OR ALTER PROCEDURE ProcedureUp3
AS
BEGIN
	CREATE TABLE Trailers(
		idTrailer SMALLINT PRIMARY KEY,
		idProduction INT,
		trailerTitle VARCHAR(50)
	)
END
GO

CREATE OR ALTER PROCEDURE ProcedureUp4
AS
BEGIN
	ALTER TABLE Trailers
	ADD trailerLength SMALLINT
END
GO

CREATE OR ALTER PROCEDURE ProcedureUp5
AS
BEGIN
	ALTER TABLE Trailers
	ADD CONSTRAINT fk_trailerIdProduction FOREIGN KEY (idProduction) REFERENCES Productions(idProduction)
END
GO

CREATE OR ALTER PROCEDURE ProcedureDown1
AS
BEGIN
	ALTER TABLE ProductionsProducers
	ALTER COLUMN salary INT
END
GO

CREATE OR ALTER PROCEDURE ProcedureDown2
AS
BEGIN
	ALTER TABLE Profiles
	DROP CONSTRAINT df_profileName
END
GO

CREATE OR ALTER PROCEDURE ProcedureDown3
AS
BEGIN
	DROP TABLE Trailers
END
GO

CREATE OR ALTER PROCEDURE ProcedureDown4
AS
BEGIN
	ALTER TABLE Trailers
	DROP COLUMN trailerLength
END
GO

CREATE OR ALTER PROCEDURE ProcedureDown5
AS
BEGIN
	ALTER TABLE Trailers
	DROP CONSTRAINT fk_traileridProduction
END
GO

CREATE OR ALTER PROCEDURE GetToVersion
@newVersion SMALLINT
AS
BEGIN
	DECLARE @currentVersion SMALLINT;
	DECLARE @procedure VARCHAR(15);
	SET @currentVersion = (SELECT currentVersion FROM DatabaseVersion);

	IF @newVersion = @currentVersion OR @newVersion > 5 OR @newVersion < 0
		RETURN;

	IF @newVersion > @currentVersion
	BEGIN
		WHILE @newVersion > @currentVersion
		BEGIN
			SET @currentVersion = @currentVersion + 1
			SET @procedure = 'ProcedureUp' + CAST(@currentVersion AS VARCHAR);
			PRINT @procedure;
			EXEC @procedure
		END;
	END;
	ELSE
	BEGIN
		WHILE @newVersion < @currentVersion
		BEGIN
			SET @procedure = 'ProcedureDown' + CAST(@currentVersion AS VARCHAR);
			SET @currentVersion = @currentVersion - 1
			PRINT @procedure;
			EXEC @procedure
		END;
	END;

	UPDATE DatabaseVersion SET currentVersion=@newVersion;
END
GO

UPDATE DatabaseVersion SET currentVersion=0
EXEC GetToVersion @newVersion=0