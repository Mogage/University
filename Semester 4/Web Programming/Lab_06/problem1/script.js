// function showDestinations(value){
//     if (value.length === 0) {
//         document.getElementById("destinations").innerHTML = "";
//         return;
//     } else {
//         let xmlhttp = new XMLHttpRequest();
//         xmlhttp.onreadystatechange = function () {
//             if (this.readyState === 4 && this.status === 200) {
//                 document.getElementById("destinations").innerHTML = this.responseText;
//             }
//         };
//         xmlhttp.open("GET", "getDestinations.php?q=" + value, true);
//         xmlhttp.send();
//     }
// }

function showDestinations(value){
    if (value.length === 0) {
        $("#destinations").html("");
        return;
    } else {
        $.ajax({
            url: "getDestinations.php?q=" + value,
            method: "GET",
            success: function(response) {
                $("#destinations").html(response);
            }
        });
    }
}