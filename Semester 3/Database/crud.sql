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

CREATE OR ALTER PROCEDURE InsertProductions
	@flag BIT OUTPUT,
	@title NVARCHAR(50),
	@type NVARCHAR(50)
AS BEGIN
	SET NOCOUNT ON;	
	IF EXISTS (SELECT * FROM Productions WHERE productionTitle = @title)
	BEGIN
		PRINT 'This production title already exists.'
		SET @flag = 0
		RETURN;
	END

	INSERT INTO Productions (productionTitle, productionType) VALUES (@title, @type)
	PRINT 'Inserted into Productions: ' + @title + ' - ' + @type
END;
GO

CREATE OR ALTER PROCEDURE DeleteProductions
	@flag BIT OUTPUT,
	@id INT
AS BEGIN
	SET NOCOUNT ON;	
	IF NOT EXISTS (SELECT * FROM Productions WHERE idProduction = @id) OR @id = NULL
	BEGIN
		PRINT 'The production with this id does not exist.'
		SET @flag = 0
		RETURN;
	END

	DELETE FROM Productions WHERE idProduction = @id
	PRINT 'Deleted from Productions the one with the id: ' + CAST(@id AS VARCHAR)
END;
GO

CREATE OR ALTER PROCEDURE UpdateProductions
	@flag BIT OUTPUT,
	@id INT,
	@title NVARCHAR(50),
	@type NVARCHAR(50)
AS BEGIN
	SET NOCOUNT ON;	
	IF NOT EXISTS (SELECT * FROM Productions WHERE idProduction = @id) OR @id = NULL
	BEGIN
		PRINT 'The production with this id does not exist.'
		SET @flag = 0
		RETURN;
	END

	UPDATE Productions SET productionTitle = @title, productionType = @type WHERE idProduction = @id
	PRINT 'Updated production with id: ' + CAST(@id AS VARCHAR)
END;
GO

CREATE OR ALTER PROCEDURE SelectProductions
	@id INT
AS BEGIN
	SET NOCOUNT ON;	
	SELECT * FROM Productions WHERE idProduction = @id
	PRINT 'Selected production with id: ' + CAST(@id AS VARCHAR)
END;
GO

CREATE OR ALTER PROCEDURE CRUDProductions
	@flag BIT OUTPUT,
	@action NVARCHAR(50),
	@id INT,
	@title NVARCHAR(50),
	@type NVARCHAR(50)
AS BEGIN
	SET NOCOUNT ON;
	SET @flag = 1

	IF dbo.StringValidation(@title, 'title') = 0 OR dbo.StringValidation(@type, 'name') = 0
	BEGIN
		SET @flag = 0
		PRINT 'Validation Error'
		RETURN;
	END

	IF @action = 'insert'
		EXEC InsertProductions @flag OUTPUT, @title, @type
	ELSE IF @action = 'delete'
		EXEC DeleteProductions @flag OUTPUT, @id
	ELSE IF @action = 'update'
		EXEC UpdateProductions @flag OUTPUT, @id, @title, @type
	ELSE IF @action = 'select'
		EXEC SelectProductions @id
	ELSE
		PRINT 'Possible actions: insert, delete, update, select.'
END;
GO

CREATE OR ALTER PROCEDURE InsertProducers
	@firstName NVARCHAR(50),
	@lastName NVARCHAR(50)
AS BEGIN
	SET NOCOUNT ON;	
	IF EXISTS (SELECT * FROM Producers WHERE producerFirstName = @firstName AND producerLastName = @lastName)
	BEGIN
		PRINT 'This producer already exists.'
		RETURN;
	END
	
	INSERT INTO Producers (producerFirstName, producerLastName) VALUES (@firstName, @lastName)
	PRINT 'Inserted into Producers: ' + @firstName + ' ' + @lastName
END;
GO

CREATE OR ALTER PROCEDURE DeleteProducers
	@id INT
AS BEGIN
	SET NOCOUNT ON;	
	IF NOT EXISTS (SELECT * FROM Producers WHERE idProducer = @id)
	BEGIN
		PRINT 'Producer with this id does not exist.'
		RETURN;
	END

	DELETE FROM Producers WHERE idProducer = @id
	PRINT 'Deleted producer with id: ' + CAST(@id AS VARCHAR)
END;
GO

CREATE OR ALTER PROCEDURE UpdateProducers
	@id INT,
	@firstName NVARCHAR(50),
	@lastName NVARCHAR(50)
AS BEGIN
	SET NOCOUNT ON;	
	IF NOT EXISTS (SELECT * FROM Producers WHERE idProducer = @id)
	BEGIN
		PRINT 'Producer with this id does not exist.'
		RETURN;
	END

	UPDATE Producers SET producerFirstName = @firstName, producerLastName = @lastName WHERE idProducer = @id
	PRINT 'Updated producer with id: ' + CAST(@id AS VARCHAR)
END;
GO

CREATE OR ALTER PROCEDURE SelectProducers
	@id INT
AS BEGIN
	SET NOCOUNT ON;	
	SELECT * FROM Producers WHERE idProducer = @id
	PRINT 'Selected producer with id: ' + CAST(@id AS VARCHAR)
END;
GO

CREATE OR ALTER PROCEDURE CRUDProducers
	@action NVARCHAR(50),
	@id INT,
	@firstName NVARCHAR(50),
	@lastName NVARCHAR(50)
AS BEGIN
	SET NOCOUNT ON;

	IF dbo.StringValidation(@firstName, 'name') = 0 OR dbo.StringValidation(@lastName, 'name') = 0
	BEGIN
		PRINT 'Validation Error'
		RETURN;
	END

	IF @action = 'insert'
		EXEC InsertProducers @firstName, @lastName
	ELSE IF @action = 'delete'
		EXEC DeleteProducers @id
	ELSE IF @action = 'update' 
		EXEC UpdateProducers @id, @firstName, @lastName
	ELSE IF @action = 'select'
		EXEC SelectProducers @id
	ELSE
		PRINT 'Possible actions: insert, delete, update, select.'
END;
GO

CREATE OR ALTER PROCEDURE InsertProductionsProducers
	@idProducer INT,
	@idProduction INT,
	@salary INT
AS BEGIN
	IF EXISTS (SELECT * FROM ProductionsProducers WHERE idProducer = @idProducer AND idProduction = @idProduction)
	BEGIN
		PRINT 'This entity already exists.'
		RETURN;
	END;

	INSERT INTO ProductionsProducers (idProducer, idProduction, salary) VALUES (@idProducer, @idProduction, @salary)
	PRINT 'Inserted entity in table with key: (' + CAST(@idProducer AS VARCHAR) + ', ' + CAST(@idProduction AS VARCHAR) + ')'
END;
GO

CREATE OR ALTER PROCEDURE DeleteProductionsProducers
	@idProducer INT,
	@idProduction INT
AS BEGIN
	SET NOCOUNT ON;	
	IF NOT EXISTS (SELECT * FROM ProductionsProducers WHERE idProducer = @idProducer AND idProduction = @idProduction)
	BEGIN
		PRINT 'This entity does not exist.'
		RETURN;
	END;

	DELETE FROM ProductionsProducers WHERE idProducer = @idProducer AND idProduction = @idProduction
	PRINT 'Deleted entity with key: (' + CAST(@idProducer AS VARCHAR) + ', ' + CAST(@idProduction AS VARCHAR) + ')'
END;
GO

CREATE OR ALTER PROCEDURE UpdateProductionsProducers
	@idProducer INT,
	@idProduction INT,
	@salary INT
AS BEGIN
	SET NOCOUNT ON;	
	IF NOT EXISTS (SELECT * FROM ProductionsProducers WHERE idProducer = @idProducer AND idProduction = @idProduction)
	BEGIN
		PRINT 'This entity does not exist.'
		RETURN;
	END;

	UPDATE ProductionsProducers set salary = @salary WHERE idProducer = @idProducer AND idProduction = @idProduction
	PRINT 'Updated entity with key: (' + CAST(@idProducer AS VARCHAR) + ', ' + CAST(@idProduction AS VARCHAR) + ')'
END;
GO

CREATE OR ALTER PROCEDURE SelectProductionsProducers
	@idProducer INT,
	@idProduction INT
AS BEGIN
	SET NOCOUNT ON;
	SELECT * FROM ProductionsProducers WHERE idProducer = @idProducer AND idProduction = @idProduction
	PRINT 'Selected on auxiliary table.'
END;
GO

CREATE OR ALTER PROCEDURE CRUDProductionsProducers
	@action NVARCHAR(50),
	@idProducer INT,
	@idProduction INT,
	@salary INT
AS BEGIN
	SET NOCOUNT ON;

	IF dbo.SalaryValidation(@salary) = 0 OR dbo.IdProducerValidation(@idProducer) = 0 OR dbo.IdProductionValidation(@idProduction) = 0
	BEGIN
		PRINT 'Validation Error'
		RETURN;
	END

	IF @action = 'insert'
		EXEC InsertProductionsProducers @idProducer, @idProduction, @salary
	ELSE IF @action = 'delete'
		EXEC DeleteProductionsProducers @idProducer, @idProduction
	ELSE IF @action = 'update' 
		EXEC UpdateProductionsProducers @idProducer, @idProduction, @salary
	ELSE IF @action = 'select'
		EXEC SelectProductionsProducers @idProducer, @idProduction
	ELSE
		PRINT 'Possible actions: insert, delete, update, select.'

END;
GO

CREATE OR ALTER PROCEDURE MainCRUD
AS BEGIN
	SET NOCOUNT ON;

	DELETE FROM Producers
	DELETE FROM Productions
	DELETE FROM ProductionsProducers

	DBCC CHECKIDENT ('Producers', RESEED, 0);
	DBCC CHECKIDENT ('Productions', RESEED, 0);

	PRINT 'Operatii pe Producers'

	EXEC CRUDProducers 'insert', 1, 'Nicolae', 'Mogage'
	EXEC CRUDProducers 'insert', 2, 'Nicolae', 'Mogage.?'
	EXEC CRUDProducers 'insert', 2, 'Nicolae/?', 'Mogage'
	EXEC CRUDProducers 'insert', 2, 'Denis', 'Moldovan'
	EXEC CRUDProducers 'insert', 3, 'Daria', 'Marian'
	EXEC CRUDProducers 'insert', 4, 'Otniel', 'Matei'
	EXEC CRUDProducers 'insert', 5, 'Eveline', 'Molnar'
	EXEC CRUDProducers 'insert', 6, 'Eveline', 'Molnar'
	EXEC CRUDProducers 'update', 4, 'Fabian', 'Martin'
	EXEC CRUDProducers 'update', 1, 'Razvan', 'Mogage'
	EXEC CRUDProducers 'delete', 1, 'Razvan', 'Mogage'
	EXEC CRUDProducers 'delete', 1, 'Razvan', 'Mogage'
	EXEC CRUDProducers 'delete', 5,  'Eveline', 'Molnar'
	EXEC CRUDProducers 'update', 1, 'Razvan', 'Mogage'
	EXEC CRUDProducers 'select', 3, 'Daria', 'Marian'
	
	-- Ids left: 2, 3 ,4

	PRINT 'Operatii pe Productions'

	DECLARE @flag BIT
	EXEC CRUDProductions @flag OUTPUT, 'insert', 1, 'Team building?', 'Movie'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'insert', 1, 'Team building', 'Movie-'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'insert', 1, 'Team building 2', 'Movie'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'insert', 2, 'The Irishman', 'Movie'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'insert', 3, 'The Irishman', 'Movie'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'insert', 3, 'Wednesday', 'Serial'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'insert', 4, 'Black list', 'Serial'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'update', 3, 'Enola Holmes 2', 'Movie'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'update', 3, 'The Irishman.?', 'Movie'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'delete', 3, 'The Irishman', 'Movie'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'update', 3, 'Enola Holmes 2', 'Movie'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'select', 2, 'The Irishman', 'Movie'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	EXEC CRUDProductions @flag OUTPUT, 'delete', 3, 'The Irishman', 'Movie'
	IF @flag = 0 PRINT 'Nu s-a intamplat nimic'
	
	-- Ids left: 1, 2 ,4

	PRINT 'Operatii pe ProductionsProducers'

	EXEC CRUDProductionsProducers 'insert', 2, 1, -10
	EXEC CRUDProductionsProducers 'insert', 2, 1, 2023
	EXEC CRUDProductionsProducers 'insert', 2, 1, 2023
	EXEC CRUDProductionsProducers 'insert', 2, 2, 2025
	EXEC CRUDProductionsProducers 'insert', 3, 2, 2026
	EXEC CRUDProductionsProducers 'insert', 4, 1, 2027
	EXEC CRUDProductionsProducers 'insert', 4, 4, 2028
	EXEC CRUDProductionsProducers 'update', 4, 4, 100
	EXEC CRUDProductionsProducers 'update', 4, 4, 2022
	EXEC CRUDProductionsProducers 'select', 4, 4, 2023
	EXEC CRUDProductionsProducers 'update', 4, 2, 2023
	EXEC CRUDProductionsProducers 'delete', 4, 4, 2023
	EXEC CRUDProductionsProducers 'update', 4, 4, 2022
	EXEC CRUDProductionsProducers 'delete', 4, 4, 2023
	EXEC CRUDProductionsProducers 'select', 4, 1, 2023

END;
GO

EXEC MainCRUD
GO

SELECT * FROM Producers;
SELECT * FROM Productions;
SELECT * FROM ProductionsProducers;
GO

CREATE INDEX IX_Producers_name_asc ON Producers (producerFirstName asc, producerLastName asc)
ALTER INDEX IX_Producers_name_asc ON Producers DISABLE
ALTER INDEX IX_Producers_name_asc ON Producers REBUILD
SELECT * FROM Producers;
SELECT * FROM ViewProducers ORDER BY producerFirstName, producerLastName;

CREATE INDEX IX_PPP_salary_desc ON ProductionsProducers (salary desc)
ALTER INDEX IX_PPP_salary_desc ON ProductionsProducers DISABLE
ALTER INDEX IX_PPP_salary_desc ON ProductionsProducers REBUILD
SELECT P.producerFirstName, P.producerLastName, PP.salary
FROM ProductionsProducers AS PP
INNER JOIN Producers AS P ON PP.idProducer = P.idProducer;
SELECT * FROM ViewPPP ORDER BY salary DESC;

SELECT i2.name, i1.user_scans, i1.user_seeks, i1.user_updates, i1.last_user_scan, i1.last_user_seek, i1.last_user_update
FROM sys.dm_db_index_usage_stats i1 INNER JOIN sys.indexes i2 ON i1.index_id = i2.index_id
WHERE  i1.object_id = i2.object_id
