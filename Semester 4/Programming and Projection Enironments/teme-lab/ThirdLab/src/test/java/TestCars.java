import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.FileReader;
import java.io.IOException;
import java.util.Properties;

public class TestCars {
    @Test
    @DisplayName("Test Cars")
    public void testExample() {
        Properties props=new Properties();
        try {
            props.load(new FileReader("bd.config"));
        } catch (IOException e) {
            System.out.println("Cannot find bd.config "+e);
        }

        CarRepository carRepo=new CarsDBRepository(props);
        Car car = new Car("Volkswagen","Passat", 2022);
        carRepo.add(car);
        System.out.println("Toate masinile din db");
        for(Car car_i:carRepo.findAll())
            System.out.println(car_i);
        System.out.println("Toate masinile de dupa 2020");
        for(Car car_i : carRepo.findBetweenYears(2020, 2023)) {
            System.out.println(car_i);
        }
        car.setManufacturer("Tesla");
        carRepo.update(10, car);
        String manufacturer="Tesla";
        System.out.println("Masinile produse de "+manufacturer);
        for(Car car_i:carRepo.findByManufacturer(manufacturer))
            System.out.println(car_i);
    }
}
