using System.Data.SQLite;
using model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data;
using persistence.databaseUtils;

namespace persistence.airports
{
    public class AirportRepository : AbstractRepository<Airport, int>, IAirportRepository
    {
        public AirportRepository(IDictionary<string, string> properties) : base(properties)
        {
            logger.Info("Initializing Airport Repository.");
        }
        protected override Airport extractEntity(IDataReader dataReader)
        {
            int id = dataReader.GetInt32(0);
            string name = dataReader.GetString(1);
            string cityName = dataReader.GetString(2);
            return new Airport(id, name, cityName);
        }

        public override int add(Airport item)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "INSERT INTO airports(name, city_name) VALUES (@name, @cityName)";

                var paramName = base.sqlCommand.CreateParameter();
                paramName.ParameterName = "@name";
                paramName.Value = item.Name;
                base.sqlCommand.Parameters.Add(paramName);

                var paramCityName = base.sqlCommand.CreateParameter();
                paramCityName.ParameterName = "@id";
                paramCityName.Value = item.CityName;
                base.sqlCommand.Parameters.Add(paramCityName);

                return base.add(item);
            }
        }

        public override void delete(Airport item)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "DELETE FROM airports WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                base.delete(item);
            }
        }
        public override void update(Airport item, int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "UPDATE airports SET name=@name, city_name=@cityName WHERE id=@id";

                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = item.ID;
                base.sqlCommand.Parameters.Add(paramId);

                var paramName = base.sqlCommand.CreateParameter();
                paramName.ParameterName = "@name";
                paramName.Value = item.Name;
                base.sqlCommand.Parameters.Add(paramName);

                var paramCityName = base.sqlCommand.CreateParameter();
                paramCityName.ParameterName = "@id";
                paramCityName.Value = item.CityName;
                base.sqlCommand.Parameters.Add(paramCityName);

                base.update(item, id);
            }
        }

        public override List<Airport> findAll()
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM airports";

                return base.findAll();
            }
        }

        public override Airport? findById(int id)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM airports WHERE id=@id";
                var paramId = base.sqlCommand.CreateParameter();
                paramId.ParameterName = "@id";
                paramId.Value = id;
                base.sqlCommand.Parameters.Add(paramId);

                return base.findById(id);
            }
        }

        public List<Airport> getAirportAfterName(string name)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM airports WHERE name=@name";

                var paramName = base.sqlCommand.CreateParameter();
                paramName.ParameterName = "@name";
                paramName.Value = name;
                base.sqlCommand.Parameters.Add(paramName);

                return base.findAll();
            }
        }

        public List<Airport> getAirportsInCity(string cityName)
        {
            base.connection = DbUtils.getConnection(base.properties);
            using (base.sqlCommand = connection.CreateCommand())
            {
                base.sqlCommand.CommandText = "SELECT * FROM airports WHERE LOWER(city_name) LIKE @cityName";
                logger.Info(base.sqlCommand.CommandText);

                var paramName = base.sqlCommand.CreateParameter();
                paramName.ParameterName = "@cityName";
                paramName.Value = cityName.ToLower() + '%';
                base.sqlCommand.Parameters.Add(paramName);

                return base.findAll();
            }
        }
    }
}
