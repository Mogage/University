using Proiect_MPP.databaseUtils;
using Proiect_MPP.domain.people;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.people.clients
{
    internal class ClientRepository : PersonRepository<Client>, IClientRepository
    {
        public ClientRepository(IDictionary<string, string> properties) : base(properties)
        {
            logger.Info("Initializing Client Repository.");
        }

        protected override Client extractEntity(IDataReader dataReader)
        {
            int id = dataReader.GetInt32(0);
            string firstName = dataReader.GetString(1);
            string lastName = dataReader.GetString(2);
            string address = dataReader.GetString(3);
            return new Client(id, firstName, lastName, address);
        }

        protected override string getTableName()
        {
            return "clients";
        }

        public override int add(Client item)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "INSERT INTO clients (id, first_name, last_name, address) VALUES (@id, @firstName, @lastName, @address)";

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

                var paramAddress = base.sqlCommand.CreateParameter();
                paramAddress.ParameterName = "@address";
                paramAddress.Value = item.Address;
                base.sqlCommand.Parameters.Add(paramAddress);

                return base.add(item);
            }
        }

        public override void delete(Client item)
        {

            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "DELETE FROM clients WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                base.delete(item);
            }
        }

        public override void update(Client item, int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "UPDATE clients SET first_name=@firstName, last_name=@lastName, address=@address WHERE id=@id";

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

                var paramAddress = base.sqlCommand.CreateParameter();
                paramAddress.ParameterName = "@address";
                paramAddress.Value = item.Address;
                base.sqlCommand.Parameters.Add(paramAddress);

                base.update(item, id);
            }
        }

        public override Client? findById(int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM clients WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = id;
                base.sqlCommand.Parameters.Add(paramId);

                return base.findById(id);
            }
        }

        public Client? findByAddress(string address)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM clients WHERE address=@address";

                var paramAddress = base.sqlCommand.CreateParameter();
                paramAddress.ParameterName = "@address";
                paramAddress.Value = address;
                base.sqlCommand.Parameters.Add(paramAddress);

                return base.getOne();
            }
        }

        public override IList<Client> findAll()
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM clients";

                return base.findAll();
            }
        }
    }
}
