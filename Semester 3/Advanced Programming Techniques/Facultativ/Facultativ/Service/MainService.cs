using Facultativ.Repository;
using Facultativ.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Service
{
    internal class MainService
    {
        GamesRepo games;
        PlayersRepo players;
        ActivePlayersRepo activePlayers;
        TeamsRepo teams;

        public MainService(
            IRepository<int, Game> games, 
            IRepository<int, Player> players,
            IRepository<int, ActivePlayer> activePlayers,
            IRepository<int, Team> teams)
        {
            this.games = (GamesRepo)games;
            this.players = (PlayersRepo)players;
            this.activePlayers = (ActivePlayersRepo)activePlayers;
            this.teams = (TeamsRepo)teams;
        }

        public List<Player> FindPlayersOfTeam(int idTeam)
        {
            return players.FindAll().Where(player => player.Team.Id == idTeam).ToList();
        }

        private bool checkActivePlayerTeam(ActivePlayer activePlayer, int idTeam, int idGame)
        {
            Player player = players.Find(activePlayer.IdPlayer);
            if (null == player ||
                player.Team.Id != idTeam ||
                activePlayer.IdGame != idGame ||
                activePlayer.Type != Utils.Constants.PlayerType.Participant)
            {
                return false;
            }
            return true;
        }

        public List<ActivePlayer> FindActivePlayersFromAGame(int idTeam, int idGame)
        {
            return activePlayers.FindAll().Where(activePlayer => checkActivePlayerTeam(activePlayer, idTeam, idGame)).ToList();
        }

        public List<Game> FindGamesInInterval(DateTime startTime, DateTime endTime)
        {
            return games.FindAll().Where(game => game.Date >= startTime && game.Date <= endTime).ToList();
        }

        public Dictionary<string, int> FindScoreOfGame(int idGame) 
        {
            Game game = games.Find(idGame);
            if (null == game) { return null; }
            Dictionary<string, int> scoresToReturn = new Dictionary<string, int> { { game.HostTeam.Name, 0 }, { game.GuestTeam.Name, 0 } };
            List<ActivePlayer> activePlayersHostTeam = FindActivePlayersFromAGame(game.HostTeam.Id, idGame);
            List<ActivePlayer> activePlayersGuestTeam = FindActivePlayersFromAGame(game.GuestTeam.Id, idGame);

            foreach (ActivePlayer activePlayer in activePlayersHostTeam)
            {
                scoresToReturn[game.HostTeam.Name] = scoresToReturn[game.HostTeam.Name] + activePlayer.ScoredPoints;
            }
            foreach (ActivePlayer activePlayer in activePlayersGuestTeam)
            {
                scoresToReturn[game.GuestTeam.Name] = scoresToReturn[game.GuestTeam.Name] + activePlayer.ScoredPoints;
            }

            return scoresToReturn;
        }

        public List<Team> GetTeams()
        {
            return teams.FindAll();
        }

        public List<Game> GetGames()
        {
            return games.FindAll();
        }

        public Player GetPlayer(int idPlayer)
        {
            return players.Find(idPlayer);
        }
    }
}
