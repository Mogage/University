// let loadIds = () => {
//     let xmlhttp = new XMLHttpRequest();
//     xmlhttp.onreadystatechange = function () {
//         if (this.readyState === 4 && this.status === 200) {
//             document.getElementById("myList").innerHTML = this.responseText;
//         }
//     };
//     xmlhttp.open("GET", "loadIds.php", true);
//     xmlhttp.send();
// }
//
// let loadForm = (id) => {
//     let xmlhttp = new XMLHttpRequest();
//     xmlhttp.onreadystatechange = function () {
//         if (this.readyState === 4 && this.status === 200) {
//             document.getElementById("myForm").innerHTML = this.responseText;
//         }
//     };
//     xmlhttp.open("GET", "loadForm.php?q=" + id, true);
//     xmlhttp.send();
// }
//
// let modifyData = () => {
//     let xmlhttp = new XMLHttpRequest();
//     xmlhttp.onreadystatechange = function () {
//         if (this.readyState === 4 && this.status === 200) {
//             document.getElementById("output").innerHTML = this.responseText;
//         }
//     };
//     xmlhttp.open("POST", "submitChanges.php", true);
//     let id = document.getElementById("id").value;
//     let nume = document.getElementById("nume").value;
//     let prenume = document.getElementById("prenume").value;
//     let email = document.getElementById("email").value;
//     let oras = document.getElementById("oras").value;
//     let strada = document.getElementById("strada").value;
//     let data = {id: id, nume: nume, prenume: prenume, email: email, oras: oras, strada: strada};
//     xmlhttp.send(JSON.stringify(data));
// }
//
// window.onload = () => {
//     document.getElementById("myForm").addEventListener("submit", (event) => {
//         event.preventDefault();
//         modifyData();
//     });
//     loadIds();
// }

let loadIds = () => {
    $.ajax({
        url: 'loadIds.php',
        type: 'GET',
        success: function(response) {
            $('#myList').html(response);
        }
    });
}

let loadForm = (id) => {
    $.ajax({
        url: 'loadForm.php?q=' + id,
        type: 'GET',
        success: function(response) {
            $('#myForm').html(response);
        }
    });
}

let modifyData = () => {
    let id = $('#id').val();
    let nume = $('#nume').val();
    let prenume = $('#prenume').val();
    let email = $('#email').val();
    let oras = $('#oras').val();
    let strada = $('#strada').val();
    let data = {id: id, nume: nume, prenume: prenume, email: email, oras: oras, strada: strada};
    $.ajax({
        url: 'submitChanges.php',
        type: 'POST',
        data: JSON.stringify(data),
        success: function(response) {
            $('#output').html(response);
        }
    });
}

$(document).ready(function() {
    $('#myForm').submit(function(event) {
        event.preventDefault();
        modifyData();
    });
    loadIds();
});