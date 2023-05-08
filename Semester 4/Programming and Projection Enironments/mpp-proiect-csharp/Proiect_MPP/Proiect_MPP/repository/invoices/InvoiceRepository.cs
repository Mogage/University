using Proiect_MPP.databaseUtils;
using Proiect_MPP.domain;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.invoices
{
    public class InvoiceRepository : AbstractRepository<Invoice, int>, IInvoiceRepository
    {
        public InvoiceRepository(IDictionary<string, string> properties) : base(properties)
        {
            logger.Info("Initializing Invoice Repository.");
        }

        protected override Invoice extractEntity(IDataReader dataReader)
        {
            int id = dataReader.GetInt32(0);
            int clientId = dataReader.GetInt32(1);
            return new Invoice(id, clientId);
        }

        public override int add(Invoice item)
        {
            base.connection = DbUtils.getConnection(base.properties);
            int id;
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "INSERT INTO invoices(client_id) VALUES (@clientId)";

                var paramClientId = base.sqlCommand.CreateParameter();
                paramClientId.ParameterName = "@clientId";
                paramClientId.Value = item.ClientId;
                base.sqlCommand.Parameters.Add(paramClientId);

                base.add(item);

                base.sqlCommand.CommandText = "select last_insert_rowid()";
                Int64 int64 = (Int64)base.sqlCommand.ExecuteScalar();
                id = Convert.ToInt32(int64);
            }

            return id;
        }

        public override void delete(Invoice item)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "DELETE FROM invoices WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                base.delete(item);
            }
        }
        public override void update(Invoice item, int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "UPDATE invoices SET client_id=@clientId WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                var paramClientId = base.sqlCommand.CreateParameter();
                paramClientId.ParameterName = "@clientId";
                paramClientId.Value = item.ClientId;
                base.sqlCommand.Parameters.Add(paramClientId);

                base.update(item, id);
            }
        }
        public override Invoice? findById(int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM invoices WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = id;
                base.sqlCommand.Parameters.Add(paramId);

                return base.findById(id);
            }
        }

        public override List<Invoice> findAll()
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM invoices";

                return base.findAll();
            }
        }
    }
}
