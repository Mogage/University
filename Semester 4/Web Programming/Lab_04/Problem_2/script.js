const form = document.querySelector('form');
const inputs = form.querySelectorAll('input');

function checkName(input) {
    return input.length >= 3;
}

function checkEmail(input) {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(input);
}

function checkPassword(input) {
    if (input.length < 8) {
        return false;
    }
    const regex = /(?=.*[A-Z])(?=.*[!@#$%^&*.])/;
    return regex.test(input);
}

function checkDate(input) {
    const regex = /[0-9]{4}\/[0-9]{2}\/[0-9]{1,2}/;
    return regex.test(input);
}

function checkInput(input, checkFunction) {

    if (input.value.trim().length === 0 || !checkFunction(input.value)) {
        input.classList.add('invalid')
    } else {
        input.classList.remove('invalid')
    }
}

form.addEventListener('submit', function(event) {
    event.preventDefault();

    checkInput(inputs.item(0), checkName);
    checkInput(inputs.item(1), checkName);
    checkInput(inputs.item(2), checkEmail);
    checkInput(inputs.item(3), checkPassword);
    checkInput(inputs.item(4), checkDate);

    if (!form.querySelector('.invalid')) {
        //form.submit();
        document.getElementById('output').innerHTML = "Datele sunt completate corect";
    } else {
        document.getElementById('output').innerHTML = "Datele nu sunt completate corect";
    }
});