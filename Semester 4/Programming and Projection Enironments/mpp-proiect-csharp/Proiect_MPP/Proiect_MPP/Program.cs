using System.Configuration;
using Proiect_MPP.domain.people;
using Proiect_MPP.repository.people.employees;
using Proiect_MPP.repository;
using log4net.Config;

[assembly: XmlConfigurator]

namespace Proiect_MPP
{
    internal static class Program
    {
        /// <summary>
        ///  The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            XmlConfigurator.Configure();
            IDictionary<string, string> props = new SortedList<string, string>();
            props.Add("ConnectionString", GetConnectionStringByName("tourismDB"));

            // To customize application configuration such as set high DPI settings or default font,
            // see https://aka.ms/applicationconfiguration.
            ApplicationConfiguration.Initialize();
            Application.Run(new LogInForm(props));
        }

        static string? GetConnectionStringByName(string name)
        {
            // Assume failure.
            string? returnValue = null;

            // Look for the name in the connectionStrings section.
            ConnectionStringSettings settings = ConfigurationManager.ConnectionStrings[name];

            // If found, return the connection string.
            if (settings != null)
                returnValue = settings.ConnectionString;

            return returnValue;
        }
    }
}