package model;

import java.io.Serializable;

public class Entity<ID> implements Serializable {
    private static final long serialVersionUID = 1231321L;

    private ID id;

    public ID getId() {
        return id;
    }

    public void setId(ID newId) {
        id = newId;
    }
}




