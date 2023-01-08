using Facultativ.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Repository
{
    internal abstract class AbstractRepo<ID, T> : IRepository<ID, T> where T : Entity<ID>
    {
        protected static List<T> entities = new List<T>();
        private string filePath;

        public AbstractRepo(string filePath)
        {
            this.filePath = filePath;
            LoadData();
        }

        public abstract T ExtractEntity(string[] values);

        protected void LoadData()
        {
            entities.Clear();

            using (StreamReader streamReader = new StreamReader(filePath))
            {
                string line;

                while ((line = streamReader.ReadLine()) != null)
                {
                    string[] values = line.Split(',');
                    T entity = ExtractEntity(values);
                    entities.Add(entity);
                }
                streamReader.Close();
            }
        }

        public static T? Find(ID id)
        {
            return entities.Find(entity => entity.Id.Equals(id));
        }

        public List<T> FindAll()
        {
            return entities;
        }

        public void Refresh()
        {
            LoadData();
        }
    }
}
