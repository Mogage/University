// window.onload = () => {
//     loadDirectory("C:/xampp/htdocs")
// }
//
// let loadDirectory = (directory) => {
//     let xmlhttp = new XMLHttpRequest();
//     xmlhttp.onreadystatechange = function () {
//         if (this.readyState === 4 && this.status === 200) {
//             document.getElementById("fileList").innerHTML = this.responseText;
//         }
//     };
//     xmlhttp.open("GET", "loadFiles.php?q=" + directory, true);
//     xmlhttp.send();
// }

$(document).ready(() => {
    loadDirectory("C:/xampp/htdocs")
});

let loadDirectory = (directory) => {
    $.ajax({
        url: "loadFiles.php",
        type: "GET",
        data: {q: directory},
        success: (response) => {
            $("#fileList").html(response);
        },
        error: (xhr, status, error) => {
            console.log(error);
        }
    });
}