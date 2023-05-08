
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
        $(input).addClass('invalid');
    } else {
        $(input).removeClass('invalid');
    }
}

$(document).ready(function () {
    const inputs = $("input");

    $("#myForm").submit(function(event) {
        event.preventDefault();

        checkInput(inputs[0], checkName);
        checkInput(inputs[1], checkName);
        checkInput(inputs[2], checkEmail);
        checkInput(inputs[3], checkPassword);
        checkInput(inputs[4], checkDate);

        if ($(".invalid").length === 0) {
            $("#output").text("Datele sunt completate corect");
        } else {
            $("#output").text("Datele nu sunt completate corect");
        }
    });
});
