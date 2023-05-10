let table = [['', '', ''], ['', '', ''], ['', '', '']];

// let putZero = (position) => {
//     table[position[0]][position[1]] = 'o';
//     document.querySelectorAll("tr").forEach((element, index) => {
//         if (index != position[0]) {
//             return true;
//         }
//         element.querySelectorAll("td").forEach((element1, index1) => {
//             if (index1 != position[1]) {
//                 return true;
//             }
//             element1.innerHTML = 'o';
//             return false;
//         })
//         return false;
//     });
// }
//
// let computerMove = () => {
//     let xmlhttp = new XMLHttpRequest();
//     xmlhttp.onreadystatechange = function () {
//         if (this.readyState === 4 && this.status === 200) {
//             if (this.responseText[this.responseText.length - 1] === "r") {
//                 setTimeout(() => {alert("Remiza")}, 1);
//             } else if (this.responseText[this.responseText.length - 1] === "o") {
//                 setTimeout(() => {alert("Ai pierdut")}, 1);
//             } else if (this.responseText[this.responseText.length - 1] === "x") {
//                 setTimeout(() => {alert("Ai castigat")}, 1);
//                 return;
//             }
//             putZero(this.responseText);
//         }
//     };
//     xmlhttp.open("POST", "computerMove.php", true);
//     xmlhttp.send(JSON.stringify({table: table}));
// }
//
// window.onload = () => {
//     if (Math.floor(Math.random() * 2) === 0) {
//         computerMove();
//     }
//
//     document.querySelectorAll('td').forEach((td, index) => {
//         td.addEventListener('click', () => {
//             if (td.innerHTML !== '') {
//                 return;
//             }
//             table[Math.floor(index / 3)][index % 3] = 'x';
//             td.innerHTML = 'x';
//
//             computerMove();
//         });
//     });
// }

let putZero = (position) => {
    table[position[0]][position[1]] = 'o';
    $("tr").each((index, element) => {
        if (index != position[0]) {
            return true;
        }
        $(element).find("td").each((index1, element1) => {
            if (index1 != position[1]) {
                return true;
            }
            $(element1).html('o');
            return false;
        })
        return false;
    });
}

let computerMove = () => {
    $.ajax({
        url: 'computerMove.php',
        type: 'POST',
        data: JSON.stringify({table: table}),
        success: function (response) {
            if (response[response.length - 1] === "r") {
                setTimeout(() => {
                    alert("Remiza")
                }, 1);
            } else if (response[response.length - 1] === "o") {
                setTimeout(() => {
                    alert("Ai pierdut")
                }, 1);
            } else if (response[response.length - 1] === "x") {
                setTimeout(() => {
                    alert("Ai castigat")
                }, 1);
                return;
            }
            putZero(response);
        }
    });
}

$(document).ready(() => {
    if (Math.floor(Math.random() * 2) === 0) {
        computerMove();
    }

    $('td').each((index, td) => {
        $(td).click(() => {
            if ($(td).html() !== '') {
                return;
            }
            table[Math.floor(index / 3)][index % 3] = 'x';
            $(td).html('x');

            computerMove();
        });
    });
});