package root;

import root.model.*;
import root.model.people.Employee;
import root.model.people.Person;

import java.time.LocalDate;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class Utils {
    public static Protobuffs.Request createLoginRequest(Employee employee) {
        Protobuffs.Employee myEmployee = Protobuffs.Employee.newBuilder().
                setId(employee.getId()).
                setFirstName(employee.getFirstName()).
                setLastName(employee.getLastName()).
                setPosition(employee.getPosition()).
                setUsername(employee.getEmail()).
                setPassword(employee.getPassword()).build();
        return Protobuffs.Request.newBuilder().setType(Protobuffs.Request.Type.LOGIN)
                .setEmployee(myEmployee).build();
    }

    public static Protobuffs.Request createLogoutRequest(Employee employee) {
        Protobuffs.Employee myEmployee = Protobuffs.Employee.newBuilder().
                setId(employee.getId()).
                setFirstName(employee.getFirstName()).
                setLastName(employee.getLastName()).
                setPosition(employee.getPosition()).
                setUsername(employee.getEmail()).
                setPassword(employee.getPassword()).build();
        return Protobuffs.Request.newBuilder().setType(Protobuffs.Request.Type.LOGOUT)
                .setEmployee(myEmployee).build();
    }

    public static Protobuffs.Request createFlightByDestinationDateRequest(DTODestinationDate destinationDate) {
        Protobuffs.DTODestinationDate myDestinationDate = Protobuffs.DTODestinationDate.newBuilder().
                setDestination(destinationDate.getDestination()).
                setDate(destinationDate.getDate().format(Constants.DATE_FORMATTER)).build();
        return Protobuffs.Request.newBuilder().setType(Protobuffs.Request.Type.GET_DD_FLIGHT)
                .setDtoDestinationDate(myDestinationDate).build();
    }

    public static Protobuffs.Request createFindAirportByIdRequest(int id) {
        return Protobuffs.Request.newBuilder().setType(Protobuffs.Request.Type.GET_AIRPORT)
                .setId(id).build();
    }

    public static Protobuffs.Request createFindFlightByIdRequest(int id) {
        return Protobuffs.Request.newBuilder().setType(Protobuffs.Request.Type.GET_FLIGHT)
                .setId(id).build();
    }

    public static Protobuffs.Request createGetAllAvailableFlightsRequest() {
        return Protobuffs.Request.newBuilder().setType(Protobuffs.Request.Type.GET_A_FLIGHTS)
                .build();
    }

    public static Protobuffs.Request createGetAllAirportsRequest() {
        return Protobuffs.Request.newBuilder().setType(Protobuffs.Request.Type.GET_ALL_AIRPORTS)
                .build();
    }

    public static Protobuffs.Request createBuyTicketRequest(DTOFlight flight) {
        Protobuffs.Client myClient = Protobuffs.Client.newBuilder().
                setId(flight.getClient().getId()).
                setFirstName(flight.getClient().getFirstName()).
                setLastName(flight.getClient().getLastName()).
                setAddress((flight.getClient().getAddress())).build();
        Protobuffs.Flight myFlight = Protobuffs.Flight.newBuilder().
                setId(flight.getFlight().getId()).
                setFreeSeats(flight.getFlight().getFreeSeats()).
                setDestinationAirport(flight.getFlight().getDestinationAirport()).
                setDepartureAirport(flight.getFlight().getDepartureAirport()).
                setDepartureDate(flight.getFlight().getDepartureDate().format(Constants.DATE_FORMATTER)).
                setDepartureTime(flight.getFlight().getDepartureTime().format(Constants.TIME_FORMATTER)).build();
        List<Protobuffs.Person> people = new ArrayList<>();
        for (Person person : flight.getPeople()) {
            Protobuffs.Person myPerson = Protobuffs.Person.newBuilder().
                    setId(person.getId()).
                    setFirstName(person.getFirstName()).
                    setLastName(person.getLastName()).build();
            people.add(myPerson);
        }
        Protobuffs.DTOFlight myDto = Protobuffs.DTOFlight.newBuilder().
                setClient(myClient).setFlight(myFlight).build();
        for (Protobuffs.Person person : people) {
            myDto = myDto.toBuilder().addPersons(person).build();
        }
        return Protobuffs.Request.newBuilder().setType(Protobuffs.Request.Type.BUY_TICKET).
                setDtoFlight(myDto).build();
    }

    public static Airport getAirport(Protobuffs.Response response) {
        Protobuffs.Airport airportResponse = response.getAirport();
        return new Airport(airportResponse.getId(), airportResponse.getName(), airportResponse.getCityName());
    }

    public static Flight getFlight(Protobuffs.Response response) {
        Protobuffs.Flight flightResponse = response.getFlight();
        return new Flight(flightResponse.getId(), flightResponse.getFreeSeats(), flightResponse.getDestinationAirport(),
                flightResponse.getDepartureAirport(), LocalDate.parse(flightResponse.getDepartureDate(), Constants.DATE_FORMATTER),
                LocalTime.parse(flightResponse.getDepartureTime(), Constants.TIME_FORMATTER));
    }

    public static Collection<Airport> getAirports(Protobuffs.Response response) {
        List<Airport> airportList = new ArrayList<>();
        for (Protobuffs.Airport airport : response.getAirportsList()) {
            Airport myAirport = new Airport(airport.getId(), airport.getName(), airport.getCityName());
            airportList.add(myAirport);
        }
        return airportList;
    }

    public static Collection<Flight> getFlights(Protobuffs.Response response) {
        List<Flight> flightList = new ArrayList<>();
        for (Protobuffs.Flight flight : response.getFlightsList()) {
            Flight myFlight = new Flight(flight.getId(), flight.getFreeSeats(), flight.getDestinationAirport(),
                    flight.getDepartureAirport(), LocalDate.parse(flight.getDepartureDate(), Constants.DATE_FORMATTER),
                    LocalTime.parse(flight.getDepartureTime(), Constants.TIME_FORMATTER));
            flightList.add(myFlight);
        }
        return flightList;
    }

    public static String getError(Protobuffs.Response response) {
        return response.getError();
    }

    public static Employee getEmployee(Protobuffs.Response response) {
        Protobuffs.Employee employeeResponse = response.getEmployee();
        return new Employee(employeeResponse.getId(), employeeResponse.getFirstName(), employeeResponse.getLastName(),
                employeeResponse.getPosition(), employeeResponse.getUsername(), employeeResponse.getPassword());
    }
}
