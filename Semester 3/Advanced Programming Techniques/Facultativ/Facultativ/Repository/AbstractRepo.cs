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
        protected List<T> entities;
        private string filePath;

        public AbstractRepo(string filePath)
        {
            entities = new List<T>();
            this.filePath = filePath;
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

        public T Find(ID id)
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
