let filters = "?";

let loadProducts = () => {
    console.log(filters);
    let xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function () {
        if (this.readyState === 4 && this.status === 200) {
            document.getElementById("produse").innerHTML = this.responseText;
        }
    };
    xmlhttp.open("GET", "loadProducts.php" + filters, true);
    xmlhttp.send();
}

window.onload = () => {
    loadProducts();
    document.querySelectorAll("input[type=checkbox]").forEach((checkbox) => {
        checkbox.addEventListener("change", () => {
            if (checkbox.checked) {
                filters += "&" + checkbox.name + "=" + checkbox.id;
            } else {
                filters = filters.replace("&" + checkbox.name + "=" + checkbox.id, "");
            }
            loadProducts();
        });
    });
}

// $(document).ready(function() {
//     loadProducts();
//     $('input[type=checkbox]').change(function() {
//         if ($(this).is(':checked')) {
//             filters += '&' + $(this).attr('name') + '=' + $(this).attr('id');
//         } else {
//             filters = filters.replace('&' + $(this).attr('name') + '=' + $(this).attr('id'), '');
//         }
//         loadProducts();
//     });
// });
//
// function loadProducts() {
//     console.log(filters);
//     $.ajax({
//         url: 'loadProducts.php' + filters,
//         type: 'GET',
//         success: function(response) {
//             $('#produse').html(response);
//         }
//     });
// }