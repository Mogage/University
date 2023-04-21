function toRun() {
    let numberPairs = [1, 2, 8, 3, 4, 7, 6, 5, 5, 8, 2, 1, 6, 7, 4, 3];
    let imagesPairs = [];
    for (let image = 1; image <= 8; image = image + 1) {
        let img = new Image();
        img.src = "images/" + image + ".jpg";
        imagesPairs.push(img);
        imagesPairs.push(img);
    }
    let index = 0;
    let selectedCells = [];

    numberPairs.sort(() => Math.random() - 0.5);
    imagesPairs.sort(() => Math.random() - 0.5);

    document.querySelectorAll("img").forEach(e => {
        //e.innerHTML = numberPairs[index];
        e.src = imagesPairs[index].src;
        index = index + 1;
    });

    document.querySelectorAll("div")
        .forEach(e => e.addEventListener("click", function () {
            //e.style.color = "#000000";
            //selectedCells.push(e);
            let img = e.querySelector(".imageCell")
            img.style.visibility = "visible";
            selectedCells.push(img);
            if (selectedCells.length === 2) {
                let value1 = selectedCells.pop();
                let value2 = selectedCells.pop();

                if (value1.src !== value2.src) {
                //if (value1.innerHTML !== value2.innerHTML) {
                    setTimeout(() => {
                        // value1.style.color = "#ffffff"
                        // value2.style.color = "#ffffff";
                        value1.style.visibility = "hidden";
                        value2.style.visibility = "hidden";
                    }, 1000);
                }
            }
        }));
}

setTimeout(toRun, 500);