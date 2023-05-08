using persistence.databaseUtils;
using model.people;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistence.people
{
    public abstract class PersonRepository<T> : AbstractRepository<T, int>, IPersonRepository<T> where T : Person
    {
        public PersonRepository(IDictionary<string, string> properties) : base(properties)
        {
            logger.Info("Initializing Person Repository.");
        }

        protected abstract string getTableName();

        public List<T> getPersonByFirstName(string firstName)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM " + getTableName() + " WHERE first_name=@firstName";

                var paramFirstName = base.sqlCommand.CreateParameter();
                paramFirstName.ParameterName = "@firstName";
                paramFirstName.Value = firstName;
                base.sqlCommand.Parameters.Add(paramFirstName);

                return base.findAll();
            }
        }

        public List<T> getPersonByLastName(string lastName)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM " + getTableName() + " WHERE last_name=@lastName";

                var paramLastName = base.sqlCommand.CreateParameter();
                paramLastName.ParameterName = "@lastName";
                paramLastName.Value = lastName;
                base.sqlCommand.Parameters.Add(paramLastName);

                return base.findAll();
            }
        }
    }
}
