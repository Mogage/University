using System;
using System.Data;
using System.Data.SqlClient;
using System.Threading;

namespace Deadlock
{
    class Program
    {
        private readonly static string connectionString = "Server=DESKTOP-P6FMEQ4;Database=Laborator_01;Integrated Security=true;TrustServerCertificate=true;";


        static void DeadlockRunner(string procedureName)
        {
            Console.WriteLine("Procedure " + procedureName + " started.");

            int stepsUntilAbort = 3;
            while (stepsUntilAbort > 0)
            {
                try
                {
                    using (var connection = new SqlConnection(connectionString))
                    {
                        using (var command = new SqlCommand(procedureName, connection))
                        {
                            command.CommandType = CommandType.StoredProcedure;
                            connection.Open();
                            command.ExecuteNonQuery();
                            stepsUntilAbort = -1;
                            Console.WriteLine("Procedure " + procedureName + " finished.");
                        }
                    }
                }
                catch (SqlException e)
                {
                    Console.WriteLine("Exception in " + procedureName);
                    if (e.Number == 1205)
                    {
                        Console.WriteLine("anything");
                        stepsUntilAbort--;
                    }
                    else
                        break;
                }
            }
            if (stepsUntilAbort == 0)
                Console.WriteLine("Procedure " + procedureName + " abandoned");
        }

        static void Deadlock1()
        {
            DeadlockRunner("deadlock1");
        }

        static void Deadlock2()
        {
            DeadlockRunner("deadlock2");
        }

        static void Main()
        {
            Thread t1 = new Thread(Deadlock1);
            Thread t2 = new Thread(Deadlock2);
            t1.Start();
            t2.Start();
        }
    }
}