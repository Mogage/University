package root.proiect_mpp.domain;

public class Airport implements Entity<Integer> {
    private int id;
    private String name;
    private String cityName;

    // Class Constructors //

    public Airport() {
        this.id = 0;
        this.name = "";
        this.cityName = "";
    }

    public Airport(String name, String cityName) {
        this.id = 0;
        this.name = name;
        this.cityName = cityName;
    }

    public Airport(int id, String name, String cityName) {
        this.id = id;
        this.name = name;
        this.cityName = cityName;
    }

    // Getters & Setters //

    @Override
    public Integer getId() {
        return id;
    }

    @Override
    public void setId(Integer id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getCityName() {
        return cityName;
    }

    public void setCityName(String cityName) {
        this.cityName = cityName;
    }

    // toString & other functions //

    @Override
    public String toString() {
        return "Airport{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", cityName='" + cityName + '\'' +
                '}';
    }
}
