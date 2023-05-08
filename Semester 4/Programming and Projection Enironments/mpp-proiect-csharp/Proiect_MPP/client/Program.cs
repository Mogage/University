using network;
using services;

namespace client
{
    internal static class Program
    {
        /// <summary>
        ///  The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            IService server = new ClientRpcProxy("127.0.0.1", 55558);
            Controller controller = new Controller(server);
            Application.Run(new LogInForm(controller));

            // ApplicationConfiguration.Initialize();
        }
    }
}