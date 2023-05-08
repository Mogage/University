using model;
using model.people;
using services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace client
{
    public class Controller : IObserver
    {
        public event EventHandler<EmployeeEventArgs> UpdateEvent;
        public IService Service { get; }
        public Employee CurrentEmployee { get; private set; }

        public Controller(IService service)
        {
            Service = service;
            CurrentEmployee = null;
        }

        public void login(string username, string password)
        {
            Employee employeeToLogIn = new Employee(username, password);
            Employee employee = Service.login(employeeToLogIn, this);
            if (null == employee)
            {
                MessageBox.Show("This email does not exists.");
                return;
            }
            if (employee.Password != password)
            {
                MessageBox.Show("Wrong password!");
                throw new Exception("Wrong password!");
            }
            CurrentEmployee = employee;
        }

        public void logout()
        {
            Service.logout(CurrentEmployee, this);
            CurrentEmployee = null;
        }

        public void ticketBought(List<Flight> flights)
        {
            OnUserEvent(new EmployeeEventArgs(EmployeeEvent.BUY_TICKET, flights));
        }

        protected virtual void OnUserEvent(EmployeeEventArgs e)
        {
            if (UpdateEvent == null) return;
            UpdateEvent(this, e);
            Console.WriteLine("Update Event called");
        }
    }
}
