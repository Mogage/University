def citire_reguli(fisier):
    with open(fisier, "r") as f:
        reguli = []
        for linie in f:
          reguli.append(linie.strip())
    return reguli

def identificare_terminale_neterminale(reguli):
    terminale = set()
    neterminale = set()
    for regula in reguli:
        partea_stanga = regula.split("->")[0]
        if partea_stanga.strip() not in neterminale:
            terminale.add(partea_stanga.strip())
        partea_dreapta = regula.split("->")[1]
        for section in partea_dreapta.split("|"):
            for simbol in section.split():
                if simbol.isupper():
                    if simbol not in neterminale:
                        neterminale.add(simbol)
                elif simbol not in terminale:
                    if simbol == ";":
                        simbol = "eps"
                    terminale.add(simbol)
    return list(terminale), list(neterminale)

if __name__ == "__main__":
    fisier = "reguli.txt"
    reguli = citire_reguli(fisier)
    terminale, neterminale = identificare_terminale_neterminale(reguli)
    simbol_start = reguli[0].split("->")[0].strip()
    terminale.sort()
    neterminale.sort()

    with open("rezultat.txt", "w") as f:
        f.write("Simbol de start\n")
        f.write(simbol_start + "\n\n")
        f.write("Neterminale\n")
        for terminal in terminale:
            f.write(terminal + "\n")
        f.write("\n")
        f.write("Terminale\n")
        for neterminal in neterminale:
            f.write(neterminal + "\n")
        f.write("\n")
        f.write("Reguli\n")
        for regula in reguli:
            f.write(regula + "\n")
