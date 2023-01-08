using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Facultativ.Utils;

namespace Facultativ.Domain
{
    internal class ActivePlayer : Entity<int>
    {
        private int idPlayer;
        private int idGame;
        private int scoredPoints;
        private Constants.PlayerType type; 

        public int IdPlayer
        {
            get { return idPlayer; }
            set { idPlayer = value; }
        }
        public int IdGame
        {
            get { return idGame; }
            set { idGame = value; }
        }
        public int ScoredPoints
        {
            get { return scoredPoints; }
            set { scoredPoints = value; }
        }
        public Constants.PlayerType Type
        {
            get { return type; }
            set { type = value; }
        }

        public ActivePlayer(int id, int idPlayer, int idGame, int scoredPoints, Constants.PlayerType type) : base(id)
        {
            this.idPlayer = idPlayer;
            this.idGame = idGame;
            this.scoredPoints = scoredPoints;
            this.type = type;
        }
        public override string ToString() 
        { 
            return "Player: " + idPlayer.ToString() + " at game: " + idGame.ToString() + " scored " + scoredPoints.ToString() + " points.";
        }
    }
}
