let index = -3;

// let loadUsers = (dataLoad) => {
//     index += dataLoad;
//     document.getElementById("previous").style.pointerEvents = (index === 0) ? "none" : "auto";
//     document.getElementById("next").style.pointerEvents = (index >= 4) ? "none" : "auto";
//     let xmlhttp = new XMLHttpRequest();
//     xmlhttp.onreadystatechange = function () {
//         if (this.readyState === 4 && this.status === 200) {
//             document.getElementById("myTable").innerHTML = this.responseText;
//         }
//     };
//     xmlhttp.open("GET", "loadUsers.php?startIndex=" + index, true);
//     xmlhttp.send();
// }
//
// window.onload = () => {
//     loadUsers(3)
// };

let loadUsers = (dataLoad) => {
    index += dataLoad;
    $("#previous").css("pointer-events", (index === 0) ? "none" : "auto");
    $("#next").css("pointer-events", (index >= 4) ? "none" : "auto");
    $.ajax({
        url: "loadUsers.php?startIndex=" + index,
        method: "GET",
        success: function(response) {
            $("#myTable").html(response);
        }
    });
}

$(document).ready(function() {
    loadUsers(3);
});
