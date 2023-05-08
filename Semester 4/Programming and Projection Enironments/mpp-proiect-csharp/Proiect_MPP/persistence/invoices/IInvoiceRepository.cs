using model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistence.invoices
{
    public interface IInvoiceRepository : Repository<Invoice, int>
    {
    }
}
