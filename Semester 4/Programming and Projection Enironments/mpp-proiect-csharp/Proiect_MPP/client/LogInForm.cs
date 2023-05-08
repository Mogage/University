using model.people;
using model;
using persistence.flights;
using persistence.people.employees;
using persistence.tickets;
using persistence;
using System.Configuration;
using services;

namespace client
{
    public partial class LogInForm : Form
    {
        private Controller controller;
        MainForm mainForm;

        public LogInForm(Controller controller)
        {
            InitializeComponent();
            this.controller = controller;
            mainForm = new MainForm(controller, this);
        }


        private void LogInForm_Load(object sender, EventArgs e)
        {
        }

        private void changeForm()
        {
            Hide();
            mainForm.Show();
        }

        private void logInButton_Click(object sender, EventArgs e)
        {
            string email = emailTextBox.Text;
            string password = passwordTextBox.Text;

            controller.login(email, password);

            emailTextBox.Text = "";
            passwordTextBox.Text = "";

            changeForm();
        }
    }
}