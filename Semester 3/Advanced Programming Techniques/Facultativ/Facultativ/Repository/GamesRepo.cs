using Facultativ.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Repository
{
    internal class GamesRepo : AbstractRepo<int, Game>
    {
        public GamesRepo(string filePath) : base(filePath)
        {}

        public override Game ExtractEntity(string[] values)
        {
            int id = Convert.ToInt32(values[0]);
            Team hostTeam = TeamsRepo.Find(Convert.ToInt32(values[1]));
            Team guestTeam = TeamsRepo.Find(Convert.ToInt32(values[2]));
            DateTime date = DateTime.Parse(values[3]);
            Game game = new Game(id, hostTeam, guestTeam, date);
            return game;
        }
    }
}
