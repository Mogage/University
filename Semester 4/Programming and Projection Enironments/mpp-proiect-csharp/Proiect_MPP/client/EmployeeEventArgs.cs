using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace client
{
    public enum EmployeeEvent
    {
        BUY_TICKET
    }

    public class EmployeeEventArgs
    {
        private readonly EmployeeEvent type;
        private readonly Object data;

        public EmployeeEventArgs(EmployeeEvent type, Object data)
        {
            this.type = type;
            this.data = data;
        }

        public EmployeeEvent Type { get => type; }
        public object Data { get => data; }
    }
}
