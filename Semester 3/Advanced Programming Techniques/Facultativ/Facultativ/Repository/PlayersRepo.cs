using Facultativ.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Repository
{
    internal class PlayersRepo : AbstractRepo<int, Player>
    {
        public PlayersRepo(string filePath) : base(filePath) { }

        public override Player ExtractEntity(string[] values)
        {
            int id = Convert.ToInt32(values[0]);
            string name = values[1];
            string school = values[2];
            Team team = TeamsRepo.Find(Convert.ToInt32(values[3]));
            Player player = new Player(id, name, school, team);
            return player;
        }
    }
}
