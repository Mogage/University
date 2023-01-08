using Facultativ.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Repository
{
    internal class TeamsRepo : AbstractRepo<int, Team>
    {
        public TeamsRepo(string filePath) : base(filePath)
        {}

        public override Team ExtractEntity(string[] values)
        {
            int id = Convert.ToInt32(values[0]);
            string name = values[1];
            Team team = new Team(id, name);
            return team;
        }
    }
}
