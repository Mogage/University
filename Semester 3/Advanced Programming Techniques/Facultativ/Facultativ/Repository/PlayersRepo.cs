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
        TeamsRepo teams;
        public PlayersRepo(string filePath, TeamsRepo teams) : base(filePath) 
        {
            this.teams = teams;
            base.LoadData();
        }   

        public override Player ExtractEntity(string[] values)
        {
            int id = Convert.ToInt32(values[0]);
            string name = values[1];
            string school = values[3];
            Team team = teams.Find(Convert.ToInt32(values[2]));
            Player player = new Player(id, name, school, team);
            return player;
        }
    }
}
