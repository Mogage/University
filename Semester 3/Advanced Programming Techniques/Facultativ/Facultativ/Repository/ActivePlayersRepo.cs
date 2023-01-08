using Facultativ.Domain;
using Facultativ.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Repository
{
    internal class ActivePlayersRepo : AbstractRepo<int, ActivePlayer>
    {
        public ActivePlayersRepo(string filePath) : base(filePath)
        {}

        public override ActivePlayer ExtractEntity(string[] values)
        {
            int id = Convert.ToInt32(values[0]);
            int idPlayer = Convert.ToInt32(values[1]);
            int idGame = Convert.ToInt32(values[2]);
            int scoredPoints = Convert.ToInt32(values[3]);
            Constants.PlayerType type;
            switch (values[4])
            {
                case "Participant":
                    type = Constants.PlayerType.Participant;
                    break;
                case "Rezerva":
                    type = Constants.PlayerType.Rezerva;
                    break;
                default:
                    break;
            }
            ActivePlayer activePlayer = new ActivePlayer(id, idPlayer, idGame, scoredPoints, type);
            return activePlayer;
        }
    }
}
