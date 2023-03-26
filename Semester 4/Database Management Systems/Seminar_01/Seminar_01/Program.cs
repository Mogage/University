using Microsoft.Data.SqlClient;

namespace ConsoleApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.Clear();
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("Hello");
            try
            {
                string connectionString = @"Server=DESKTOP-P6FMEQ4;Database=seminar01;Integrated Security=true;TrustServerCertificate=true";
                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    Console.WriteLine("Starea conexiunii: {0}", connection.State);
                    connection.Open();
                    Console.WriteLine("Starea conexiunii: {0}", connection.State);

                    SqlCommand sqlCommand = new SqlCommand("INSERT INTO Flori(nume, pret, sezon) VALUES (@nume, @pret, @sezon);", connection);
                    sqlCommand.Parameters.AddWithValue("@nume", "lalea");
                    sqlCommand.Parameters.AddWithValue("@pret", 7.0F);
                    sqlCommand.Parameters.AddWithValue("@sezon", "primavara");
                    int rowCount = sqlCommand.ExecuteNonQuery();
                    Console.WriteLine("Row count affected: {0}", rowCount);
                    SqlCommand sqlMartisorCommand = new SqlCommand("INSERT INTO Martisoare(nume, pret, model) VALUES (@nume, @pret, @model);", connection);
                    sqlMartisorCommand.Parameters.AddWithValue("@nume", "cosar");
                    sqlMartisorCommand.Parameters.AddWithValue("@pret", 50.0F);
                    sqlMartisorCommand.Parameters.AddWithValue("@model", "scump");
                    rowCount = sqlMartisorCommand.ExecuteNonQuery();
                    Console.WriteLine("Row count affected: {0}", rowCount);


                    connection.Close();
                }
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(e.ToString());
            }
        }
    }
}