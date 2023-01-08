using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Domain
{
    internal class Student : Entity<int>
    {
        private string name;
        private string school;
        public string Name 
        { 
            get { return name; }
            set { name = value; }
        }
        public string School 
        {
            get { return school; }
            set { school = value; }
        }
        public Student(int id, string name, string school) : base(id)
        {
            this.name = name;
            this.school = school;
        }
        public override string ToString()
        {
            return name;
        }
    }
}
