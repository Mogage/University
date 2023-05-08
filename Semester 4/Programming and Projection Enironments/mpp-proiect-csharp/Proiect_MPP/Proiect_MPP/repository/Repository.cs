using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository
{
    public interface Repository<T, Tid>
    {
        int add(T item);
        void delete(T item);
        void update(T item, Tid id);
        T? findById(Tid id);
        List<T> findAll();
    }
}
