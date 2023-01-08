using Facultativ.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Repository
{
    internal interface IRepository<ID, T> where T : Entity<ID>
    {
        extern T? Find(ID id);

        List<T> FindAll();

        void Refresh();
    }
}
