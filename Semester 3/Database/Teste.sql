USE Laborator_01
GO

CREATE OR ALTER PROCEDURE InsertDataToTable
@tableName NVARCHAR(50), @itemsNumber INT
AS
BEGIN
	DECLARE @index INT, @productionID INT, @producerID INT, @userID INT
	DECLARE @profileID INT, @ratingValue INT, @maxValue INT
	SET @index = 0;

	SET NOCOUNT ON;
	IF @tableName = 'Users'
	BEGIN
		WHILE @index < @itemsNumber
		BEGIN
			INSERT INTO Users (userEmail, userPassword) VALUES ('prenume.nume@', 'parola123')
			SET @index = @index + 1
		END
	END
	ELSE IF @tableName = 'Profiles'
	BEGIN
		SET @userId = (SELECT TOP (1) [idUser] FROM [Laborator_01].[dbo].[Users])
		WHILE @index < @itemsNumber
		BEGIN
			INSERT INTO Profiles (idUser, profileName) VALUES (@userID, 'profileName')
			SET @index = @index + 1
			IF @index % 5 = 0
				SET @userID = @userID + 1
		END
	END
	ELSE IF @tableName = 'Ratings'
	BEGIN
		SET @maxValue = (SELECT TOP (1) [idProduction] FROM [Laborator_01].[dbo].[Productions] ORDER BY [idProduction] DESC)
		SET @profileID = (SELECT TOP (1) [idProfile] FROM [Laborator_01].[dbo].[Profiles])
		SET @productionID = (SELECT TOP (1) [idProduction] FROM [Laborator_01].[dbo].[Productions])
		SET @ratingValue = 0
		WHILE @index < @itemsNumber
		BEGIN
			SET @ratingValue = @ratingValue + 1
			SET @productionID = @productionID + 2
			INSERT INTO Ratings (idProfile, idProduction, ratingValue) VALUES (@profileID, @productionID, @ratingValue)
			SET @index = @index + 1
			IF @ratingValue = 10
				SET @ratingValue = 0
			IF @index % 5 = 0
				SET @profileID = @profileID + 1
			IF @productionID > @maxValue - 2
				SET @productionID = (SELECT TOP (1) [idProduction] FROM [Laborator_01].[dbo].[Productions]) + 1
		END
	END
	ELSE IF @tableName = 'Productions'
	BEGIN
		WHILE @index < @itemsNumber
		BEGIN
			INSERT INTO Productions (productionTitle, productionType) VALUES ('numeProductie', 'tipProductie')
			SET @index = @index + 1
		END
	END
	ELSE IF @tableName = 'Producers'
	BEGIN
		WHILE @index < @itemsNumber
		BEGIN
			INSERT INTO Producers (producerFirstName, producerLastName) VALUES ('prenume', 'nume')
			SET @index = @index + 1
		END
	END
	ELSE IF @tableName = 'ProductionsProducers'
	BEGIN
		SET @maxValue = (SELECT TOP (1) [idProduction] FROM [Laborator_01].[dbo].[Productions] ORDER BY [idProduction] DESC)
		SET @productionID = (SELECT TOP (1) [idProduction] FROM [Laborator_01].[dbo].[Productions])
		SET @producerID = (SELECT TOP (1) [idProducer] FROM [Laborator_01].[dbo].[Producers])
		WHILE @index < @itemsNumber
		BEGIN
			SET @productionID = @productionID + 2
			INSERT INTO ProductionsProducers (idProduction, idProducer) VALUES (@productionID, @producerID)
			SET @index = @index + 1
			IF @index % 5 = 0
				SET @producerID = @producerID + 1
			IF @productionID > @maxValue - 2
				SET @productionID = (SELECT TOP (1) [idProduction] FROM [Laborator_01].[dbo].[Productions]) + 1
		END
	END
END


CREATE OR ALTER PROCEDURE TestDatabase
AS
BEGIN
	SET NOCOUNT ON
	DECLARE @IDTest INT, @NoOfRows INT, @IDTable INT, @testRunId INT
	DECLARE @TableName NVARCHAR(50), @TestName NVARCHAR(50)
	DECLARE @testStartTime DATETIME, @testFinishTime DATETIME
	DECLARE @auxStartTime DATETIME, @auxFinishTime DATETIME

	DECLARE CursorTest CURSOR FORWARD_ONLY 
	FOR SELECT * FROM Tests
	OPEN CursorTest
	FETCH NEXT FROM CursorTest INTO @IDTest, @TestName

	DELETE FROM TestRuns;
	DELETE FROM TestRunTables;
	DELETE FROM TestRunViews;
	
	WHILE @@FETCH_STATUS = 0 BEGIN
		PRINT 'TEST NR -> ' + cast(@IDTest AS VARCHAR)

		DECLARE CursorDelete CURSOR SCROLL
		FOR SELECT T.Name, TT.NoOfRows, TT.TableID
			FROM Tables T INNER JOIN TestTables TT ON T.TableID = TT.TableID
			WHERE TestID = @IDTest
			ORDER BY Position
		OPEN CursorDelete

		FETCH NEXT FROM CursorDelete INTO @TableName, @NoOfRows, @IDTable
		WHILE @@FETCH_STATUS = 0 BEGIN
			EXEC ('DELETE FROM ' + @TableName)
			FETCH NEXT FROM CursorDelete INTO @TableName, @NoOfRows, @IDTable
		END

		SET @testStartTime = SYSDATETIME()
		INSERT INTO TestRuns (Description, StartAt)
		VALUES (@TestName, @testStartTime)
		SET @testRunId = @@IDENTITY;


		FETCH PRIOR FROM CursorDelete INTO @TableName, @NoOfRows, @IDTable
		WHILE @@FETCH_STATUS = 0 BEGIN
			SET @auxStartTime = SYSDATETIME()
			EXEC InsertDataToTable @tableName = @TableName, @itemsNumber=@NoOfRows
			SET @auxFinishTime = SYSDATETIME()

			INSERT INTO TestRunTables (TestRunID, TableID, StartAt, EndAt)
			VALUES (@testRunId, @IDTable, @auxStartTime, @auxFinishTime)

			FETCH PRIOR FROM CursorDelete INTO @TableName, @NoOfRows, @IDTable
		END
		CLOSE CursorDelete
		DEALLOCATE CursorDelete

		DECLARE @IDView INT, @ViewName NVARCHAR(50);
		DECLARE CursorView CURSOR FORWARD_ONLY
		FOR SELECT TV.ViewID, V.Name
			FROM TestViews TV INNER JOIN Views V ON TV.ViewID = V.ViewID
			WHERE TV.TestID = @IDTest;

		OPEN CursorView
		FETCH NEXT FROM CursorView INTO @IDView, @ViewName;

		WHILE @@FETCH_STATUS = 0 BEGIN
			SET @auxStartTime = SYSDATETIME()
			EXEC ('SELECT * FROM ' + @ViewName)
			SET @auxFinishTime = SYSDATETIME()

			INSERT INTO TestRunViews(TestRunID, ViewID, StartAt, EndAt)
			VALUES (@testRunId, @IDView, @auxStartTime, @auxFinishTime)

			FETCH NEXT FROM CursorView INTO @IDView, @ViewName;
		END

		CLOSE CursorView
		DEALLOCATE CursorView

		SET @testFinishTime = SYSDATETIME()
		UPDATE TestRuns SET EndAt=@testFinishTime where TestRunID=@TestRunId;

		FETCH NEXT FROM CursorTest INTO @IDTest, @TestName
	END

	CLOSE CursorTest
	DEALLOCATE CursorTest

END
GO

EXEC TestDatabase;
