using log4net;
using Proiect_MPP.databaseUtils;
using Proiect_MPP.domain.people;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.people.employees
{
    public class EmployeeRepository : PersonRepository<Employee>, IEmployeeRepository
    {

        public EmployeeRepository(IDictionary<string, string> properties) : base(properties)
        {
            logger.Info("Initializing Employee Repository.");
        }

        protected override string getTableName()
        {
            return "employees";
        }

        protected override Employee extractEntity(IDataReader dataReader)
        {
            int id = dataReader.GetInt32(0);
            string firstName = dataReader.GetString(1);
            string lastName = dataReader.GetString(2);
            string position = dataReader.GetString(3);
            string email = dataReader.GetString(4);
            string password = dataReader.GetString(5);
            return new Employee(id, firstName, lastName, position, email, password);
        }

        public override int add(Employee item)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "INSERT INTO employees (first_name, last_name, position, email, password)" +
                    " VALUES (@firstName, @lastName, @position, @email, @password)";

                var paramFirstName = base.sqlCommand.CreateParameter();
                paramFirstName.ParameterName = "@firstName";
                paramFirstName.Value = item.FirstName;
                base.sqlCommand.Parameters.Add(paramFirstName);

                var paramLastName = base.sqlCommand.CreateParameter();
                paramLastName.ParameterName = "@lastName";
                paramLastName.Value = item.LastName;
                base.sqlCommand.Parameters.Add(paramLastName);

                var paramPosition = base.sqlCommand.CreateParameter();
                paramPosition.ParameterName = "@position";
                paramPosition.Value = item.Position;
                base.sqlCommand.Parameters.Add(paramPosition);

                var paramEmail = base.sqlCommand.CreateParameter();
                paramEmail.ParameterName = "@email";
                paramEmail.Value = item.Email;
                base.sqlCommand.Parameters.Add(paramEmail);

                var paramPassword = base.sqlCommand.CreateParameter();
                paramPassword.ParameterName = "@password";
                paramPassword.Value = item.Password;
                base.sqlCommand.Parameters.Add(paramPassword);

                return base.add(item);
            }
        }

        public override void delete(Employee item)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "DELETE FROM employees WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                base.delete(item);
            }
        }

        public override void update(Employee item, int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "UPDATE employees SET first_name=@firstName, last_name=@lastName, position=@position, " +
                    "email=@email, password=@password WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                var paramFirstName = base.sqlCommand.CreateParameter();
                paramFirstName.ParameterName = "@firstName";
                paramFirstName.Value = item.FirstName;
                base.sqlCommand.Parameters.Add(paramFirstName);

                var paramLastName = base.sqlCommand.CreateParameter();
                paramLastName.ParameterName = "@lastName";
                paramLastName.Value = item.LastName;
                base.sqlCommand.Parameters.Add(paramLastName);

                var paramPosition = base.sqlCommand.CreateParameter();
                paramPosition.ParameterName = "@position";
                paramPosition.Value = item.Position;
                base.sqlCommand.Parameters.Add(paramPosition);

                var paramEmail = base.sqlCommand.CreateParameter();
                paramEmail.ParameterName = "@email";
                paramEmail.Value = item.Email;
                base.sqlCommand.Parameters.Add(paramEmail);

                var paramPassword = base.sqlCommand.CreateParameter();
                paramPassword.ParameterName = "@password";
                paramPassword.Value = item.Password;
                base.sqlCommand.Parameters.Add(paramPassword);

                base.update(item, id);
            }
        }

        public override Employee? findById(int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM employees WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = id;
                base.sqlCommand.Parameters.Add(paramId);

                return base.findById(id);
            }
        }

        public Employee? findByEmail(string email)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM employees WHERE email=@email";

                var paramEmail = base.sqlCommand.CreateParameter();
                paramEmail.ParameterName = "@email";
                paramEmail.Value = email;
                base.sqlCommand.Parameters.Add(paramEmail);

                return base.getOne();
            }
        }

        public List<Employee> getByPosition(string position)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM employees WHERE position=@position";

                var paramPosition = base.sqlCommand.CreateParameter();
                paramPosition.ParameterName = "@position";
                paramPosition.Value = position;
                base.sqlCommand.Parameters.Add(paramPosition);

                return base.findAll();
            }
        }

        public override List<Employee> findAll()
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM employees";

                return base.findAll();
            }
        }
    }
}
