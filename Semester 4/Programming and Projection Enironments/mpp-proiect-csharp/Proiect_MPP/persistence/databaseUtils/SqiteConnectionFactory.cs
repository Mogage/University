using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistence.databaseUtils
{
    public class SqiteConnectionFactory : ConnectionFactory
    {
        public override IDbConnection createConnection(IDictionary<string, string> props)
        {
            //Mono Sqlite Connection

            //String connectionString = "URI=file:/Users/grigo/didactic/MPP/ExempleCurs/2017/database/tasks.db,Version=3";
            String connectionString = props["ConnectionString"];
            Console.WriteLine("SQLite ---Se deschide o conexiune la  ... {0}", connectionString);
            return new SQLiteConnection(connectionString);

            // Windows SQLite Connection, fisierul .db ar trebuie sa fie in directorul debug/bin
            //String connectionString = "Data Source=tasks.db;Version=3";
            //return new SQLiteConnection(connectionString);
        }
    }
}
