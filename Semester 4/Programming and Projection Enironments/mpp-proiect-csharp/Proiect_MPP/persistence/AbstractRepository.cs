using System.Data.SQLite;
using model;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using log4net;

namespace persistence
{
    public abstract class AbstractRepository<T, Tid> : Repository<T, Tid> where T : Entity<Tid>
    {
        protected IDictionary<string, string> properties;
        protected IDbConnection? connection;
        protected IDbCommand? sqlCommand;
        protected static readonly ILog logger = LogManager.GetLogger("AbstractRepository");

        // Class constructors //

        public AbstractRepository(IDictionary<string, string> properties)
        {
            logger.Info("Initializing Abstract Repository.");
            this.properties = properties;
            this.connection = null;
            this.sqlCommand = null;
        }

        protected abstract T extractEntity(IDataReader dataReader);

        private void executeCommand()
        {
            try
            {
                int result = sqlCommand.ExecuteNonQuery();
                logger.InfoFormat("Modified {0} instances", result);
            }
            catch (Exception ex)
            {
                logger.Error(ex);
            }
        }

        private List<T> executeQuery()
        {
            List<T> elements = new List<T> { };
            try
            {
                using (var dataReader = sqlCommand.ExecuteReader()) 
                {
                    while (dataReader.Read())
                    {
                        elements.Add(extractEntity(dataReader));
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Error(ex);
            }
            return elements;
        }

        public virtual int add(T item)
        {
            logger.InfoFormat("Saving entity {0}", item);
            executeCommand();
            return 0;
        }

        public virtual void delete(T item)
        {
            logger.InfoFormat("Deleting entity {0}", item);
            executeCommand();
        }

        public virtual void update(T item, Tid id)
        {
            logger.InfoFormat("Updating entity {0}", item);
            executeCommand();
        }
        public virtual T? findById(Tid id)
        {
            logger.InfoFormat("Finding entity after id {0}", id);
            return getOne();
        }

        public virtual List<T> findAll()
        {
            logger.Info("Finding all elements");
            return executeQuery();
        }

        public T? getOne()
        {
            List<T> elements = executeQuery();
            if (elements.Count() == 0)
            {
                return null;
            }
            return elements[0];
        }
    }
}
