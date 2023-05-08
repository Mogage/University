let slideIndex = 0;
let timer;

function toRun() {
    showSlides(slideIndex);
    slideIndex++;
    timer = setTimeout(toRun, 3000);
}

function plusSlides(n) {
    showSlides(slideIndex += n);
    clearTimeout(timer);
    timer = setTimeout(toRun, 3000);
}

function showSlides(n) {
    let slides = $(".mySlides");

    if (n >= slides.length) {
        slideIndex = 0;
    }
    if (n < 0) {
        slideIndex = slides.length - 1;
    }

    slides.each((i, slide) => {
        // $(slide).css("display", "none");
        $(slide).hide();
    });

    // $(slides[slideIndex]).css("display", "block");
    $(slides[slideIndex]).show();
}

$(document).ready(() => {
    toRun();
});