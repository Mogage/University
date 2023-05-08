using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace model
{
    [Serializable]
    public class Invoice : Entity<int>
    {
        private int clientId;

        // Class Constructors //

        public Invoice() : base(0)
        {
            this.clientId = 0;
        }

        public Invoice(int clientId) : base(0)
        {
            this.clientId = clientId;
        }

        public Invoice(int id, int clientId) : base(id)
        {
            this.clientId = clientId;
        }

        // Getters & Setters //

        public int ClientId
        {
            get { return this.clientId; }
            set { this.clientId = value; }
        }

        // toString & other functions //

        public override string ToString()
        {
            return "Invoice{" +
                "id=" + base.ID +
                ", clientId=" + clientId +
                '}';
        }
    }
}
