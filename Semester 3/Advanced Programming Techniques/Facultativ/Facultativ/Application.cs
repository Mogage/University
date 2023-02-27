using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Facultativ.Repository;
using Facultativ.Domain;
using Facultativ.Service;
using Facultativ.User_Interface;

namespace Facultativ
{
    class Application
    {
        private static void RunApp()
        {
            string gamesRepoFilePath = "F:\\Git_repo\\University\\Semester 3\\Advanced Programming Techniques\\Facultativ\\Facultativ\\Data\\games.txt";
            string playersRepoFilePath = "F:\\Git_repo\\University\\Semester 3\\Advanced Programming Techniques\\Facultativ\\Facultativ\\Data\\players.txt";
            string activePlayersRepoFilePath = "F:\\Git_repo\\University\\Semester 3\\Advanced Programming Techniques\\Facultativ\\Facultativ\\Data\\activePlayers.txt";
            string teamsRepoFilePath = "F:\\Git_repo\\University\\Semester 3\\Advanced Programming Techniques\\Facultativ\\Facultativ\\Data\\teams.txt";

            IRepository<int, Team> teams = new TeamsRepo(teamsRepoFilePath);
            IRepository<int, Game> games = new GamesRepo(gamesRepoFilePath, (TeamsRepo)teams);
            IRepository<int, Player> players = new PlayersRepo(playersRepoFilePath, (TeamsRepo)teams);
            IRepository<int, ActivePlayer> activePlayers = new ActivePlayersRepo(activePlayersRepoFilePath);

            MainService service = new MainService(games, players, activePlayers, teams);

            UserInterface userInterface = new UserInterface(service);
            userInterface.Run();
        }

        public static void Main(String[] Args)
        {
            RunApp();
        }
    }
}
