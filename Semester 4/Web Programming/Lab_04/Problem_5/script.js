let slideIndex = 0;
let timer;


function toRun() {
    showSlides(slideIndex);
    slideIndex++;
    timer = setTimeout(toRun, 5000);
}

function plusSlides(n) {
    showSlides(slideIndex += n);
    clearTimeout(timer);
    timer = setTimeout(toRun, 5000);
}

function showSlides(n) {
    let i;
    let slides = document.getElementsByClassName("mySlides");
    if (n >= slides.length) {
        slideIndex = 0;
    }
    if (n < 0) {
        slideIndex = slides.length - 1;
    }
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }

    slides[slideIndex].style.display = "block";
}

setTimeout(toRun, 500);