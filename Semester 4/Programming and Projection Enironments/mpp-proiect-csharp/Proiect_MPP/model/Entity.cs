using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace model
{
    [Serializable]
    public abstract class Entity<Tid>
    {
        private Tid id;

        // Class Construcors //

        public Entity(Tid id)
        {
            this.id = id;
        }

        // Getters & Setters //

        public Tid ID
        {
            get { return this.id; }
            set { this.id = value; }
        }
    }
}
