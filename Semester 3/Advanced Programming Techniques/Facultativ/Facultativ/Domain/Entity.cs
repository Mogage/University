using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Domain
{
    internal class Entity<ID>
    {
        private ID id;
        public Entity(ID id) => this.id = id;
        public ID Id
        {
            get { return this.id; }
            set { this.id = value; }
        }
        public object Nume { get; internal set; } = default!;

        public override string ToString()
        {
            if (this.id == null) { return ""; }
            return id.ToString() + " ";
        }
    }
}
