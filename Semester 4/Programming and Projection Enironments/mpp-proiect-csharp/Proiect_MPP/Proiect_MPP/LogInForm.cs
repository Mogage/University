using Proiect_MPP.domain.people;
using Proiect_MPP.domain;
using Proiect_MPP.repository.flights;
using Proiect_MPP.repository.people.employees;
using Proiect_MPP.repository.tickets;
using Proiect_MPP.repository;
using System.Configuration;
using Proiect_MPP.service.logIn;

namespace Proiect_MPP
{
    public partial class LogInForm : Form
    {
        private ILogInService logInService;
        IDictionary<string, string> properties;
        MainForm mainForm;

        public LogInForm(IDictionary<string, string> properties)
        {
            InitializeComponent();
            this.properties = properties;
            logInService = new LogInService(new EmployeeRepository(properties));
            mainForm = new MainForm(properties, this);
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

            Employee employee = logInService.findByEmail(email);

            if (null == employee)
            {
                MessageBox.Show("This email does not exists.");
                return;
            }

            if (employee.Password != password)
            {
                MessageBox.Show("Wrong password!");
                return;
            }
            emailTextBox.Text = "";
            passwordTextBox.Text = "";

            changeForm();
        }
    }
}