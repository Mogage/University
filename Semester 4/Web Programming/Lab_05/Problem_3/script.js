$(document).ready(function () {
    let numberPairs = [1, 2, 8, 3, 4, 7, 6, 5, 5, 8, 2, 1, 6, 7, 4, 3];
    let imagesPairs = [];
    for (let image = 1; image <= 8; image = image + 1) {
        let img = new Image();
        img.src = "images/" + image + ".jpg";
        imagesPairs.push(img);
        imagesPairs.push(img);
    }
    let selectedCells = [];

    numberPairs.sort(() => Math.random() - 0.5);
    imagesPairs.sort(() => Math.random() - 0.5);

    $("div").each((index, e) => {
        $(e).html(numberPairs[index]);
        // e.src = imagesPairs[index].src
        $(e).click(() => {
            // images[index].style.color = "#000000";
            $(e).css("color", "#000000");
            selectedCells.push(e);
            // let img = e.querySelector(".imageCell")
            // img.style.visibility = "visible";
            // selectedCells.push(img);
            if (selectedCells.length === 2) {
                let value1 = selectedCells.pop();
                let value2 = selectedCells.pop();

                // if (value1.src !== value2.src) {
                if ($(value1).html() !== $(value2).html()) {
                    setTimeout(() => {
                        $(value1).css("color", "#ffffff");
                        $(value2).css("color", "#ffffff");
                        // value1.style.visibility = "hidden";
                        // value2.style.visibility = "hidden";
                    }, 1000);
                }
            }
        });
    });
});