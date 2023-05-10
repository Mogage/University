USE Laborator_01;
GO

CREATE OR ALTER FUNCTION StringValidation (@string NVARCHAR(50), @stringType NVARCHAR(50))
RETURNS BIT
AS BEGIN
	DECLARE @length INT
	DECLARE @count INT
	DECLARE @characters NVARCHAR(50)

	SET @length = LEN(@string)
	SET @count = 1
	IF @stringType = 'name'
		SET @characters = '%[A-Za-z ]'
		IF NOT LEFT(@string, 1) = UPPER(LEFT(@string, 1)) Collate SQL_Latin1_General_CP1_CS_AS 
			RETURN 0;
	ELSE
		SET @characters = '%[A-Za-z 0-9]'

	IF @string IS NULL
		RETURN 0;
	IF LTRIM(@string) = ''
		RETURN 0;
	IF LEN(@string) < 3
		RETURN 0;

	WHILE @count <= @length 
	BEGIN
		IF NOT SUBSTRING(@string, @count, 1) LIKE @characters
			RETURN 0;
		SET @count = @count + 1
	END;

	RETURN 1;
END;
GO

CREATE OR ALTER FUNCTION SalaryValidation (@salary INT)
RETURNS BIT
AS BEGIN
	IF @salary < 1000
		RETURN 0;

	RETURN 1;
END;
GO

CREATE OR ALTER FUNCTION IdProductionValidation (@id INT)
RETURNS BIT
AS BEGIN
	IF NOT EXISTS (SELECT * FROM Productions WHERE idProduction = @id)
		RETURN 0;

	RETURN 1;
END;
GO

CREATE OR ALTER FUNCTION IdProducerValidation (@id INT)
RETURNS BIT
AS BEGIN
	IF NOT EXISTS (SELECT * FROM Producers WHERE idProducer = @id)
		RETURN 0;

	RETURN 1;
END;
GO

CREATE OR ALTER FUNCTION ValidateProducer (
	@firstName NVARCHAR(50),
	@lastName NVARCHAR(50)
)
RETURNS VARCHAR(255)
AS BEGIN

	DECLARE @error VARCHAR(255)
	SET @error = ''

	IF (dbo.StringValidation(@firstName, 'name') = 0)
	BEGIN
		SET @error = @error + 'First name invalid.';
	END
	IF (dbo.StringValidation(@lastName, 'name') = 0)
	BEGIN
		SET @error = @error + 'Last name invalid';
	END

	RETURN @error;
END;
GO

CREATE OR ALTER FUNCTION ValidateProduction (
	@title NVARCHAR(50),
	@type NVARCHAR(50)
)
RETURNS VARCHAR(255)
AS BEGIN

	DECLARE @error VARCHAR(255)
	SET @error = ''

	IF (dbo.StringValidation(@title, 'title') = 0)
	BEGIN
		SET @error = @error + 'Title invalid.';
	END
	IF (dbo.StringValidation(@type, 'name') = 0)
	BEGIN
		SET @error = @error + 'Type invalid';
	END

	RETURN @error;
END;
GO

CREATE OR ALTER FUNCTION ValidateProducerProduction (
	@salary INT,
	@idProducer INT,
	@idProduction INT

)
RETURNS VARCHAR(255)
AS BEGIN

	DECLARE @error VARCHAR(255)
	SET @error = ''

	IF (dbo.SalaryValidation(@salary) = 0)
	BEGIN
		SET @error = @error + 'Salary invalid.';
	END
	IF (dbo.IdProducerValidation(@idProducer) = 0)
	BEGIN
		SET @error = @error + 'Id producer invalid.';
	END
	IF (dbo.IdProductionValidation(@idProduction) = 0)
	BEGIN
		SET @error = @error + 'Id production invalid.';
	END

	RETURN @error;
END;
GO

CREATE TABLE LOGS(
	table_name VARCHAR(50),
	realisation_time DATETIME
)
GO;

DROP TABLE LOGS

CREATE OR ALTER PROCEDURE InsertProductionProducersV1
	@firstName NVARCHAR(50),
	@lastName NVARCHAR(50),
	@title NVARCHAR(50),
	@type NVARCHAR(50),
	@salary INT
AS BEGIN
	BEGIN TRAN 
	BEGIN TRY
		DECLARE @error VARCHAR(255)
		SET @error = dbo.ValidateProducer(@firstName, @lastName);
		IF (@error != '')
		BEGIN
			PRINT @error
			RAISERROR (@error, 14, 1)
		END

		INSERT INTO Producers(producerFirstName, producerLastName) VALUES (@firstName, @lastName)
		INSERT INTO Logs(table_name, realisation_time) VALUES ('Producers', CURRENT_TIMESTAMP)

		SET @error = dbo.ValidateProduction(@title, @type);
		IF (@error != '')
		BEGIN
			PRINT @error
			RAISERROR (@error, 14, 1)
		END

		INSERT INTO Productions(productionTitle, productionType) VALUES (@title, @type)
		INSERT INTO Logs(table_name, realisation_time) VALUES ('Productions', CURRENT_TIMESTAMP)

		DECLARE @idProducer INT
		DECLARE @idProduction INT

		SET @idProducer = (SELECT idProducer FROM Producers WHERE producerFirstName = @firstName AND producerLastName = @lastName);
		SET @idProduction = (SELECT idProduction FROM Productions WHERE productionTitle = @title AND productionType = @type);

		SET @error = dbo.ValidateProducerProduction(@salary, @idProducer, @idProduction);
		IF (@error != '')
		BEGIN
			PRINT @error
			RAISERROR (@error, 14, 1)
		END

		INSERT INTO ProductionsProducers(idProducer, idProduction, salary) VALUES (@idProducer, @idProduction, @salary);
		INSERT INTO Logs(table_name, realisation_time) VALUES ('ProducersProductions', CURRENT_TIMESTAMP)
		
		COMMIT TRAN
		SELECT 'Commit'
	END TRY
	BEGIN CATCH
		ROLLBACK TRAN
		INSERT INTO Logs(table_name, realisation_time) VALUES ('Rollback Trans', CURRENT_TIMESTAMP)
		SELECT 'Rollback'
	END CATCH
END;
GO

select * from Producers
select * from Productions
select * from ProductionsProducers
select * from Logs
GO;

EXEC dbo.InsertProductionProducersV1 'Nicolae', 'Mogage', 'Interstellar', 'Movie', '2000'

select * from Producers
select * from Productions
select * from ProductionsProducers
select * from Logs
GO;

EXEC dbo.InsertProductionProducersV1 'Nicolae2', 'Mogage2', 'Interstellar2', 'Movie', '500'

select * from Producers
select * from Productions
select * from ProductionsProducers
select * from Logs
GO;

CREATE OR ALTER PROCEDURE InsertProductionProducersV2
	@firstName NVARCHAR(50),
	@lastName NVARCHAR(50),
	@title NVARCHAR(50),
	@type NVARCHAR(50),
	@salary INT
AS BEGIN
	DECLARE @error VARCHAR(255)

	BEGIN TRAN 
	BEGIN TRY
		SET @error = dbo.ValidateProducer(@firstName, @lastName);
		PRINT @firstName
		PRINT @error
		IF (@error != '')
		BEGIN
			PRINT @error
			RAISERROR (@error, 14, 1)
		END

		INSERT INTO Producers(producerFirstName, producerLastName) VALUES (@firstName, @lastName)
		INSERT INTO Logs(table_name, realisation_time) VALUES ('Producers', CURRENT_TIMESTAMP)

		COMMIT TRAN
		SELECT 'Commit producers'
	END TRY
	BEGIN CATCH
		ROLLBACK
		INSERT INTO Logs(table_name, realisation_time) VALUES ('Rollback Trans Producers', CURRENT_TIMESTAMP)
		SELECT 'Rollback producers'
	END CATCH

	
	BEGIN TRAN 
	BEGIN TRY
		SET @error = dbo.ValidateProduction(@title, @type);
		IF (@error != '')
		BEGIN
			PRINT @error
			RAISERROR (@error, 14, 1)
		END

		INSERT INTO Productions(productionTitle, productionType) VALUES (@title, @type)
		INSERT INTO Logs(table_name, realisation_time) VALUES ('Productions', CURRENT_TIMESTAMP)

		COMMIT TRAN
		SELECT 'Commit productions'
	END TRY
	BEGIN CATCH
		ROLLBACK
		INSERT INTO Logs(table_name, realisation_time) VALUES ('Rollback Trans Productions', CURRENT_TIMESTAMP)
		SELECT 'Rollback productions'
	END CATCH

	BEGIN TRAN 
	BEGIN TRY
		DECLARE @idProducer INT
		DECLARE @idProduction INT

		SET @idProducer = (SELECT idProducer FROM Producers WHERE producerFirstName = @firstName AND producerLastName = @lastName);
		SET @idProduction = (SELECT idProduction FROM Productions WHERE productionTitle = @title AND productionType = @type);

		SET @error = dbo.ValidateProducerProduction(@salary, @idProducer, @idProduction);
		IF (@error != '')
		BEGIN
			PRINT @error
			RAISERROR (@error, 14, 1)
		END

		INSERT INTO ProductionsProducers(idProducer, idProduction, salary) VALUES (@idProducer, @idProduction, @salary);
		INSERT INTO Logs(table_name, realisation_time) VALUES ('ProducersProductions', CURRENT_TIMESTAMP)
		
		COMMIT TRAN
		SELECT 'Commit ProducersProductions'
	END TRY
	BEGIN CATCH
		ROLLBACK
		INSERT INTO Logs(table_name, realisation_time) VALUES ('Rollback Trans ProducersProductions', CURRENT_TIMESTAMP)
		SELECT 'Rollback ProducersProductions'
	END CATCH

END;
GO

select * from Producers
select * from Productions
select * from ProductionsProducers
select * from Logs
GO;

EXEC dbo.InsertProductionProducersV2 'Fabian', 'Martin', 'Team building', 'Movie', '2000'

select * from Producers
select * from Productions
select * from ProductionsProducers
select * from Logs
GO;

EXEC dbo.InsertProductionProducersV2 'Denis', 'Moldovan', 'Horror', '', '500'

select * from Producers
select * from Productions
select * from ProductionsProducers
select * from Logs
GO;

EXEC dbo.InsertProductionProducersV2 'Denis Angel', 'Moldovan', 'Horror', 'Movie', '500'

select * from Producers
select * from Productions
select * from ProductionsProducers
select * from Logs
GO;

EXEC dbo.InsertProductionProducersV2 'Denis32', 'Mo', 'Horror', '', '500'

select * from Producers
select * from Productions
select * from ProductionsProducers
select * from Logs
GO;

DELETE FROM ProductionsProducers WHERE idProducer = 12
DELETE FROM ProductionsProducers WHERE idProducer = 15

DELETE FROM Productions WHERE productionTitle = 'Interstellar'
DELETE FROM Productions WHERE productionTitle = 'Team Building'
DELETE FROM Productions WHERE productionTitle = 'Horror'

DELETE FROM Producers WHERE producerFirstName = 'Nicolae' AND producerLastName = 'Mogage'
DELETE FROM Producers WHERE producerFirstName = 'Fabian' AND producerLastName = 'Martin'
DELETE FROM Producers WHERE producerFirstName = 'Denis' AND producerLastName = 'Moldovan'
DELETE FROM Producers WHERE producerFirstName = 'Denis Angel' AND producerLastName = 'Moldovan'
DELETE FROM Producers WHERE producerFirstName = 'Denis32' AND producerLastName = 'Moldovan3'