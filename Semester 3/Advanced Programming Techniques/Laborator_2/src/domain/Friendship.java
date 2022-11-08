package domain;

public class Friendship {
    private final long idUser1;
    private final long idUser2;

    public Friendship(long idUser1, long idUser2) {
        this.idUser1 = idUser1;
        this.idUser2 = idUser2;
    }

    public long getIdUser1() {
        return idUser1;
    }

    public long getIdUser2() {
        return idUser2;
    }
}
