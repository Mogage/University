using Facultativ.Domain;
using Facultativ.Repository;
using Facultativ.Service;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.User_Interface
{
    internal class UserInterface
    {
        MainService service;

        public UserInterface(MainService service)
        {
            this.service = service;
        }

        public void Run()
        {
            string userInput;
            ShowMenu();
            while(true)
            {
                Console.Write(">>");
                userInput = Console.ReadLine();
                switch(userInput) 
                {
                    case "":
                        continue;
                    case "0":
                        return;
                    case "1":
                        ShowMenu();
                        break;
                    case "2":
                        DoTask1();
                        break;
                    case "3":
                        DoTask2();
                        break;
                    case "4":
                        DoTask3();
                        break;
                    case "5":
                        DoTask4();
                        break;
                    default:
                        Console.WriteLine("This command does not exist.");
                        break;
                }
            }
        }
        private void ShowMenu()
        {
            Console.Write("0 -> Exit.\n" +
                "1 -> Afiseaza meniu.\n" +
                "2 -> Sa se afiseze toti jucatorii unei echipe data\n" +
                "3 -> Sa se afiseze toti jucatorii activi ai unei echipe de la un anumit meci\n" +
                "4 -> Sa se afiseze toate meciurile dintr-o anumita perioada calendaristica\n" +
                "5 -> Sa se determine si sa se afiseze scorul de la un anumit meci\n"
                );
        }

        private void PrintEntities<T>(List<T> entities)
        {
            int length = entities.Count();
            for (int index = 0; index < length; index = index + 1)
            {
                Console.Write((index + 1) + "\t");
                Console.WriteLine(entities[index]);
            }
        }

        private void PrintTeams(List<Team> teams)
        {
            Console.WriteLine("Choose team to find all players:");
            PrintEntities(teams);
        }

        private void PrintGames(List<Game> games)
        {
            Console.WriteLine("Choose game to find all active players");
            PrintEntities(games);
        }

        private T ReadFromUser<T>(List<T> entities)
        {
            Console.Write(">>");
            string userInput = Console.ReadLine();
            try
            {
                int position = Convert.ToInt32(userInput);
                return entities[position - 1];
            }
            catch (Exception)
            {
                Console.WriteLine("Only numerical numbers are accepted.");
            }
            return default(T);
        }

        private Team ReadTeamFromUser()
        {
            List<Team> teams = service.GetTeams();
            PrintTeams(teams);
            return ReadFromUser(teams);
        }

        private Game ReadGameFromUser(List<Game> games)
        {
            PrintGames(games);
            return ReadFromUser(games);
        }

        private void DoTask1() 
        {
            Team team = ReadTeamFromUser();
            if (null == team)
            {
                Console.WriteLine("A team couldn't be selected.\n");
                return;
            }
            List<Player> players = service.FindPlayersOfTeam(team.Id);
            foreach (Player player in players) 
            {
                Console.Write('\t');
                Console.WriteLine(player);
            }
        }

        private void DoTask2() 
        {
            Team team = ReadTeamFromUser();
            if (null == team)
            {
                Console.WriteLine("A team couldn't be selected.\n");
                return;
            }
            List<Game> games = service.GetGames().Where(game => game.HostTeam.Id == team.Id || game.GuestTeam.Id == team.Id).ToList();
            if (0 == games.Count())
            {
                Console.WriteLine("This team doesn't have any games.");
                return;
            }
            Game game = ReadGameFromUser(games);
            if(null == game)
            {
                Console.WriteLine("A team couldn't be selected.\n");
                return;
            }
            List<ActivePlayer> activePlayers = service.FindActivePlayersFromAGame(team.Id, game.Id);
            Player player = null;
            foreach (ActivePlayer activePlayer in activePlayers)
            {
                player = service.GetPlayer(activePlayer.IdPlayer);
                Console.Write('\t');
                Console.WriteLine(player);
            }
        }

        private void DoTask3()
        {
            Console.WriteLine("Introdu perioada in care sa vezi meciurile (ZZ-LL-AAAA / ZZ.LL.AAA)");
            Console.Write("Start date: ");
            string startTime = Console.ReadLine();
            Console.Write("End date: ");
            string endTime = Console.ReadLine();
            List<Game> games = service.FindGamesInInterval(DateTime.ParseExact(startTime, "dd.MM.yyyy", CultureInfo.InvariantCulture), DateTime.ParseExact(endTime, "dd.MM.yyyy", CultureInfo.InvariantCulture));
            foreach (Game game in games) 
            {
                Console.Write('\t');
                Console.WriteLine(game);
            }
        }

        private void DoTask4() 
        {
            Game game = ReadGameFromUser(service.GetGames());
            if (null == game)
            {
                Console.WriteLine("A team couldn't be selected.\n");
                return;
            }
            Dictionary<string, int> scores = service.FindScoreOfGame(game.Id);
            Console.Write($"{scores.Keys.ToArray()[0]} - {scores.Values.ToArray()[0]} : " +
                $"{scores.Values.ToArray()[1]} - {scores.Keys.ToArray()[1]}\n");
        }
    }
}
