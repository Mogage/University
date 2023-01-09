using Facultativ.Domain;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Repository
{
    internal class GamesRepo : AbstractRepo<int, Game>
    {
        TeamsRepo teams;
        public GamesRepo(string filePath, TeamsRepo teams) : base(filePath)
        {
            this.teams = teams;
            base.LoadData();
        }

        public override Game ExtractEntity(string[] values)
        {
            int id = Convert.ToInt32(values[0]);
            Team hostTeam = this.teams.Find(Convert.ToInt32(values[1]));
            Team guestTeam = this.teams.Find(Convert.ToInt32(values[2]));
            DateTime date = DateTime.ParseExact(values[3], "dd.MM.yyyy", CultureInfo.InvariantCulture);
            Game game = new Game(id, hostTeam, guestTeam, date);
            return game;
        }
    }
}
