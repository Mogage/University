using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistence.databaseUtils
{
    public class DbUtils
    {
        private static IDbConnection? instance = null;

        public static IDbConnection getConnection(IDictionary<string, string> props)
        {
            if (instance == null || instance.State == ConnectionState.Closed)
            {
                instance = getNewConnection(props);
                instance.Open();
            }
            return instance;
        }

        private static IDbConnection getNewConnection(IDictionary<string, string> props)
        {
            return ConnectionFactory.getInstance().createConnection(props);
        }
    }
}
