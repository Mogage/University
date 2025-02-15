
-- Dirty reads wrong

SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

BEGIN TRAN

	SELECT * FROM  Productions;
	INSERT INTO Logs VALUES('productions - drw', CURRENT_TIMESTAMP)
	WAITFOR DELAY '00:00:03'

	SELECT * FROM  Productions;
	INSERT INTO Logs VALUES('productions - drw', CURRENT_TIMESTAMP)

COMMIT TRAN

-- Dirty reads fix

SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

BEGIN TRAN

	SELECT * FROM  Productions;
	INSERT INTO Logs VALUES('productions - drf', CURRENT_TIMESTAMP)
	WAITFOR DELAY '00:00:05'

	SELECT * FROM  Productions;
	INSERT INTO Logs VALUES('productions - drf', CURRENT_TIMESTAMP)

COMMIT TRAN


-- Unrepeatable reads wrong

SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

BEGIN TRAN

SELECT * FROM Productions
INSERT INTO Logs VALUES('productions - urw', CURRENT_TIMESTAMP)
WAITFOR DELAY '00:00:05'
SELECT * FROM Productions
INSERT INTO Logs VALUES('productions - urw', CURRENT_TIMESTAMP)

COMMIT TRAN

-- Unrepeatable reads fix

SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

BEGIN TRAN

SELECT * FROM Productions
INSERT INTO Logs VALUES('productions - urf', CURRENT_TIMESTAMP)
WAITFOR DELAY '00:00:05'
SELECT * FROM Productions
INSERT INTO Logs VALUES('productions - urf', CURRENT_TIMESTAMP)

COMMIT TRAN

-- Phantom reads wrong

SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

BEGIN TRAN

SELECT * FROM Productions WHERE idProduction BETWEEN 1 AND 1050;
INSERT INTO Logs VALUES('Productions - prw', CURRENT_TIMESTAMP);
WAITFOR DELAY '00:00:06'
SELECT * FROM Productions WHERE idProduction BETWEEN 1 AND 1050;
INSERT INTO Logs VALUES('Productions - prw', CURRENT_TIMESTAMP);

COMMIT TRAN

-- Phantom reads fix

SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

BEGIN TRAN

SELECT * FROM Productions WHERE idProduction BETWEEN 1 AND 100;
INSERT INTO Logs VALUES('Productions - prf', CURRENT_TIMESTAMP);
WAITFOR DELAY '00:00:06'
SELECT * FROM Productions WHERE idProduction BETWEEN 1 AND 100;
INSERT INTO Logs VALUES('Productions - prf', CURRENT_TIMESTAMP);

COMMIT TRAN

-- deadlock

SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SET DEADLOCK_PRIORITY HIGH;

BEGIN TRY
	BEGIN TRAN;

	UPDATE Producers SET producerFirstName = 'deadlock name 2' WHERE idProducer=6
	INSERT INTO Logs VALUES('Productions - dl 2', CURRENT_TIMESTAMP);
	WAITFOR DELAY '00:00:05'
	UPDATE Productions SET productionTitle = 'deadlock name 2' WHERE idProduction=13
	INSERT INTO Logs VALUES('Producers - dl 2', CURRENT_TIMESTAMP);

	COMMIT TRAN;
END TRY
BEGIN CATCH
	IF ERROR_NUMBER() = 1205
	BEGIN
		ROLLBACK;
		BEGIN TRAN;

		PRINT 'Retry deadlock'

		INSERT INTO Logs VALUES('Productions - dl 2', CURRENT_TIMESTAMP);
		INSERT INTO Logs VALUES('Producers - dl 2', CURRENT_TIMESTAMP);

		COMMIT TRAN;
	END
END CATCH

SELECT * FROM Producers
SELECT * FROM Productions

GO;