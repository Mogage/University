
-- Dirty reads

BEGIN TRAN

UPDATE Productions SET productionTitle = 'movieName' WHERE idProduction = 15
INSERT INTO Logs VALUES('Productions - dr', CURRENT_TIMESTAMP);
WAITFOR DELAY '00:00:05'

ROLLBACK TRAN

-- Unrepeatable reads

DELETE FROM Productions WHERE productionTitle = 'serialName2'
INSERT INTO Productions(productionTitle, productionType) VALUES('serialName', 'serial')

BEGIN TRAN

WAITFOR DELAY '00:00:03'
UPDATE Productions SET productionTitle = 'serialName2' WHERE productionTitle='serialName'
INSERT INTO Logs VALUES('Productions - ur', CURRENT_TIMESTAMP);

COMMIT TRAN

-- Phantom reads

DELETE FROM Productions WHERE productionTitle = 'serialNou'

BEGIN TRAN

WAITFOR DELAY '00:00:03'
INSERT INTO Productions(productionTitle, productionType) VALUES ('serialNou', 'serial')
INSERT INTO Logs VALUES('Productions - pr', CURRENT_TIMESTAMP);

COMMIT TRAN

-- deadlock

SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

BEGIN TRY
	BEGIN TRAN;

	UPDATE Productions SET productionTitle = 'deadlock name 1' WHERE idProduction=13
	INSERT INTO Logs VALUES('Productions - dl 2', CURRENT_TIMESTAMP);
	WAITFOR DELAY '00:00:05'
	UPDATE Producers SET producerFirstName = 'deadlock name 1' WHERE idProducer=6
	INSERT INTO Logs VALUES('Producers - dl 2', CURRENT_TIMESTAMP);

	COMMIT TRAN;
END TRY
BEGIN CATCH
	IF ERROR_NUMBER() = 1205
	BEGIN
		ROLLBACK;
		BEGIN TRAN;

		PRINT 'Retry deadlock'

		UPDATE Productions SET productionTitle = 'deadlock name 1' WHERE idProduction=13
		INSERT INTO Logs VALUES('Productions - dl 2', CURRENT_TIMESTAMP);
		WAITFOR DELAY '00:00:03'
		UPDATE Producers SET producerFirstName = 'deadlock name 1' WHERE idProducer=6
		INSERT INTO Logs VALUES('Producers - dl 2', CURRENT_TIMESTAMP);

		COMMIT TRAN;
	END
END CATCH

SELECT * FROM Producers
SELECT * FROM Productions

GO;


CREATE OR ALTER PROCEDURE deadlock1
AS
BEGIN
	SET DEADLOCK_PRIORITY HIGH
	BEGIN TRAN
		UPDATE Productions SET productionTitle = 'deadlock name' WHERE idProduction=13
		WAITFOR DELAY '00:00:03'
		UPDATE Producers SET producerFirstName = 'deadlock name' WHERE idProducer=6
	COMMIT TRAN
END
GO;

CREATE OR ALTER PROCEDURE deadlock2
AS
BEGIN
	SET DEADLOCK_PRIORITY HIGH
	BEGIN TRAN
		UPDATE Producers SET producerFirstName = 'deadlock name' WHERE idProducer=6
		WAITFOR DELAY '00:00:03'
		UPDATE Productions SET productionTitle = 'deadlock name' WHERE idProduction=13		
	COMMIT TRAN
END
GO;