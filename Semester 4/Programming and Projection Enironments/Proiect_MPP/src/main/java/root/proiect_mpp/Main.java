package root.proiect_mpp;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;
import root.proiect_mpp.domain.Flight;
import root.proiect_mpp.domain.Ticket;
import root.proiect_mpp.domain.people.Employee;
import root.proiect_mpp.repositories.Repository;
import root.proiect_mpp.repositories.flights.FlightRepository;
import root.proiect_mpp.repositories.people.employees.EmployeeRepository;
import root.proiect_mpp.repositories.tickets.TicketRepository;

import java.io.FileReader;
import java.io.IOException;
import java.util.Properties;

public class Main extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        Properties props = new Properties();
        props.load(new FileReader("bd.config"));
        Repository<Employee, Integer> employeeRepository = new EmployeeRepository(props);
        Repository<Ticket, Integer> ticketRepository = new TicketRepository(props);
        Repository<Flight, Integer> flightRepository = new FlightRepository(props);

        Employee employee = new Employee("nicu", "mog", "seller", "mog@yahoo.com", "salut");
        employeeRepository.add(employee);
        FXMLLoader fxmlLoader = new FXMLLoader(Main.class.getResource("mainView.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 320, 240);
        stage.setTitle("Hello!");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}