using Proiect_MPP.databaseUtils;
using Proiect_MPP.domain;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.flights
{
    public class FlightRepository : AbstractRepository<Flight, int>, IFlightRepository
    {
        public FlightRepository(IDictionary<string, string> properties) : base(properties)
        {
            logger.Info("Initializing Flight Repository.");
        }

        protected override Flight extractEntity(IDataReader dataReader)
        {
            int id = dataReader.GetInt32(0);
            int freeSeats = dataReader.GetInt32(1);
            int destinationAirport = dataReader.GetInt32(2);
            int departureAirport = dataReader.GetInt32(3);
            DateTime departureDate = DateTime.ParseExact(dataReader.GetString(4), "dd/MM/yyyy");
            DateTime departureTime = DateTime.Parse(dataReader.GetString(5));
            return new Flight(id, freeSeats, destinationAirport, departureAirport, departureDate, departureTime);
        }

        public override int add(Flight item)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "INSERT INTO flights(id, free_seats, destination_airport, departure_airport, departure_date, departure_time)" +
                    "VALUES (@id, @freeSeats, @destinationAirport, @departureAirport, @departureDate, @departureTime)";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                var paramSeats = base.sqlCommand.CreateParameter();
                paramSeats.ParameterName = "@freeSeats";
                paramSeats.Value = item.FreeSeats;
                base.sqlCommand.Parameters.Add(paramSeats);

                var paramDestinationAirport = base.sqlCommand.CreateParameter();
                paramDestinationAirport.ParameterName = "@destinationAirport";
                paramDestinationAirport.Value = item.DestinationAirport;
                base.sqlCommand.Parameters.Add(paramDestinationAirport);

                var paramDepartureAirport = base.sqlCommand.CreateParameter();
                paramDepartureAirport.ParameterName = "@departureAiport";
                paramDepartureAirport.Value = item.DepartureAirport;
                base.sqlCommand.Parameters.Add(paramDepartureAirport);

                var paramDate = base.sqlCommand.CreateParameter();
                paramDate.ParameterName = "@departureDate";
                paramDate.Value = item.DepartureDate.ToString("dd-MM-yyyy");
                base.sqlCommand.Parameters.Add(paramDate);

                var paramTime = base.sqlCommand.CreateParameter();
                paramTime.ParameterName = "@departureTime";
                paramTime.Value = item.DepartureTime.ToString("hh:mm:ss");
                base.sqlCommand.Parameters.Add(paramTime);

                return base.add(item);
            }
        }

        public override void delete(Flight item)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "UPDATE flights SET free_seats=@freeSeats, destination_airport=@destinationAirport, departure_airport=@departureAirport, " +
                    "departure_date=@departureDate, departure_time=@departureTime WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                base.delete(item);
            }
        }

        public override void update(Flight item, int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "UPDATE flights SET free_seats=@freeSeats, destination_airport=@destinationAirport, departure_airport=@departureAirport, " +
                    "departure_date=@departureDate, departure_time=@departureTime WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                var paramSeats = base.sqlCommand.CreateParameter();
                paramSeats.ParameterName = "@freeSeats";
                paramSeats.Value = item.FreeSeats;
                base.sqlCommand.Parameters.Add(paramSeats);

                var paramDestinationAirport = base.sqlCommand.CreateParameter();
                paramDestinationAirport.ParameterName = "@destinationAirport";
                paramDestinationAirport.Value = item.DestinationAirport;
                base.sqlCommand.Parameters.Add(paramDestinationAirport);

                var paramDepartureAirport = base.sqlCommand.CreateParameter();
                paramDepartureAirport.ParameterName = "@departureAirport";
                paramDepartureAirport.Value = item.DepartureAirport;
                base.sqlCommand.Parameters.Add(paramDepartureAirport);

                var paramDate = base.sqlCommand.CreateParameter();
                paramDate.ParameterName = "@departureDate";
                paramDate.Value = item.DepartureDate.ToString("dd/MM/yyyy");
                base.sqlCommand.Parameters.Add(paramDate);

                var paramTime = base.sqlCommand.CreateParameter();
                paramTime.ParameterName = "@departureTime";
                paramTime.Value = item.DepartureTime.ToString("HH:mm");
                base.sqlCommand.Parameters.Add(paramTime);

                base.update(item, id);
            }
        }

        public override Flight? findById(int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM flights WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = id;
                base.sqlCommand.Parameters.Add(paramId);

                return base.findById(id);
            }
        }

        public override List<Flight> findAll()
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM flights";

                return base.findAll();
            }
        }
        public List<Flight> getAfterDepartureDateTime(DateTime departureDateTime)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM flights WHERE departure_date>=@departureDate AND departure_time>=@departureTime";

                var paramDate = base.sqlCommand.CreateParameter();
                paramDate.ParameterName = "@departureDate";
                paramDate.Value = departureDateTime.ToString("dd-MM-yyyy");
                base.sqlCommand.Parameters.Add(paramDate);

                var paramTime = base.sqlCommand.CreateParameter();
                paramTime.ParameterName = "@departureTime";
                paramTime.Value = departureDateTime.ToString("hh:mm:ss");
                base.sqlCommand.Parameters.Add(paramTime);

                return base.findAll();
            }
        }

        public List<Flight> getByDepartureAirport(int departureAirport)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM flights WHERE departure_airport=@departureAirport";

                var paramDepartureAirport = base.sqlCommand.CreateParameter();
                paramDepartureAirport.ParameterName = "@departureAiport";
                paramDepartureAirport.Value = departureAirport;
                base.sqlCommand.Parameters.Add(paramDepartureAirport);

                return base.findAll();
            }
        }

        public List<Flight> getByDestinationAirport(int destinationAirport)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM flights WHERE destination_airport=@destinationAirport";

                var paramDestinationAirport = base.sqlCommand.CreateParameter();
                paramDestinationAirport.ParameterName = "@destinationAirport";
                paramDestinationAirport.Value = destinationAirport;
                base.sqlCommand.Parameters.Add(paramDestinationAirport);

                return base.findAll();
            }
        }

        public List<Flight> getAvailable()
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM flights WHERE free_seats <> 0";
                return base.findAll();
            }
        }
    }
}
