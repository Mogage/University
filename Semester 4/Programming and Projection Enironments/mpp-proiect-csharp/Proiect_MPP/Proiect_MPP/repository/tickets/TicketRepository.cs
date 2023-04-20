using log4net.Repository.Hierarchy;
using Proiect_MPP.databaseUtils;
using Proiect_MPP.domain;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;

namespace Proiect_MPP.repository.tickets
{
    internal class TicketRepository : AbstractRepository<Ticket, int>, ITicketRepository
    {
        public TicketRepository(IDictionary<string, string> properties) : base(properties)
        {
            logger.Info("Initializing Ticket Repository.");
        }

        protected override Ticket extractEntity(IDataReader dataReader)
        {
            int id = dataReader.GetInt32(0);
            int flightId = dataReader.GetInt32(1);
            int seatNumber = dataReader.GetInt32(2);
            string touristName = dataReader.GetString(3);
            int invoiceId = dataReader.GetInt32(4);
            return new Ticket(id, flightId, invoiceId, seatNumber, touristName);
        }

        public override int add(Ticket item)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "INSERT INTO tickets (flight_id, invoice_id, seat_number, tourist_name)" +
                    " VALUES (@flightId, @invoiceId, @seatNumber, @touristName)";

                var paramFlightId = base.sqlCommand.CreateParameter();
                paramFlightId.ParameterName = "@flightId";
                paramFlightId.Value = item.FlightId;
                base.sqlCommand.Parameters.Add(paramFlightId);

                var paramInvoiceId = base.sqlCommand.CreateParameter();
                paramInvoiceId.ParameterName = "@invoiceId";
                paramInvoiceId.Value = item.InvoiceId;
                base.sqlCommand.Parameters.Add(paramInvoiceId);

                var paramSeatNumber = base.sqlCommand.CreateParameter();
                paramSeatNumber.ParameterName = "@seatNumber";
                paramSeatNumber.Value = item.SeatNumber;
                base.sqlCommand.Parameters.Add(paramSeatNumber);

                var paramTouristName = base.sqlCommand.CreateParameter();
                paramTouristName.ParameterName = "@touristName";
                paramTouristName.Value = item.TouristName;
                base.sqlCommand.Parameters.Add(paramTouristName);

                return base.add(item);
            }
        }

        public override void delete(Ticket item)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "DELETE FROM tickets WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                base.delete(item);
            }
        }

        public override void update(Ticket item, int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "UPDATE tickets SET flight_id=@flightId, invoice_id=@invoiceId, seat_number=@seatNumber, tourist_name=@touristName" +
                    " WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                var paramFlightId = base.sqlCommand.CreateParameter();
                paramFlightId.ParameterName = "@flightId";
                paramFlightId.Value = item.FlightId;
                base.sqlCommand.Parameters.Add(paramFlightId);

                var paramInvoiceId = base.sqlCommand.CreateParameter();
                paramInvoiceId.ParameterName = "@invoiceId";
                paramInvoiceId.Value = item.InvoiceId;
                base.sqlCommand.Parameters.Add(paramInvoiceId);

                var paramSeatNumber = base.sqlCommand.CreateParameter();
                paramSeatNumber.ParameterName = "@seatNumber";
                paramSeatNumber.Value = item.SeatNumber;
                base.sqlCommand.Parameters.Add(paramSeatNumber);

                var paramTouristName = base.sqlCommand.CreateParameter();
                paramTouristName.ParameterName = "@touristName";
                paramTouristName.Value = item.TouristName;
                base.sqlCommand.Parameters.Add(paramTouristName);

                base.update(item, id);
            }
        }

        public override Ticket? findById(int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM tickets WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = id;
                base.sqlCommand.Parameters.Add(paramId);

                return base.findById(id);
            }
        }
        public Ticket? findByTouristName(string name)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM tickets WHERE tourist_name=@name";

                var paramName = base.sqlCommand.CreateParameter();
                paramName.ParameterName = "@name";
                paramName.Value = name;
                base.sqlCommand.Parameters.Add(paramName);

                return base.getOne();
            }
        }

        public override IList<Ticket> findAll()
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM tickets";

                return base.findAll();
            }
        }

        public IEnumerable<Ticket> getByFlightId(int flightId)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM tickets WHERE flight_id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = flightId;
                base.sqlCommand.Parameters.Add(paramId);

                return base.findAll();
            }
        }

        public IEnumerable<Ticket> getByInvoiceId(int invoiceId)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM tickets WHERE invoice_id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = invoiceId;
                base.sqlCommand.Parameters.Add(paramId);

                return base.findAll();
            }
        }
    }
}
