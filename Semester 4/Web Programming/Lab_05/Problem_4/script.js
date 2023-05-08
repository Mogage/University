function ascending(a, b) {
    if (a.sortValue < b.sortValue) {
        return -1;
    }
    if (a.sortValue > b.sortValue) {
        return 1;
    }
    return 0;
}

function descending(a, b) {
    if (a.sortValue > b.sortValue) {
        return -1;
    }
    if (a.sortValue < b.sortValue) {
        return 1;
    }
    return 0;
}

function sortElementsLines(index, rows) {
    let toSort = [];

    rows.each((index1, element1) => {
        if (index1 === 0) {
            return true;
        }

        let toAdd = [$(element1).html()];
        toAdd.sortValue = $($(element1).find("div")[index]).html();
        toSort.push(toAdd);
    });

    let before = toSort.toString();
    if (!isNaN(parseInt(toSort[0].sortValue))) {
        for (let index = 0; index < toSort.length; index++) {
            toSort[index].sortValue = parseInt(toSort[index].sortValue);
        }
    }

    toSort.sort(ascending);
    if (before === toSort.toString()) {
        toSort.sort(descending);
    }

    for (let index = 0; index < rows.length - 1; index++) {
        $(rows[index + 1]).html(toSort[index]);
    }
}

function sortElementsCols(colIndex, rows) {
    let toSort = [];

    for (let index = 0; index < $(rows[0]).children().length - 1; index++) {
        toSort.push([]);
    }

    rows.each((index1, element1) => {
        let divs = $(element1).find("div");
        for (let nthChild = 1; nthChild < divs.length; nthChild++) {
            toSort[nthChild - 1].push($(divs[nthChild]).html());

            if (index1 !== colIndex) {
                continue;
            }
            toSort[nthChild - 1].sortValue = $(divs[nthChild]).html();
        }
    });

    let before = toSort.toString();
    if (!isNaN(parseInt(toSort[0].sortValue))) {
        for (let index = 0; index < toSort.length; index++) {
            toSort[index].sortValue = parseInt(toSort[index].sortValue);
        }
    }

    toSort.sort(ascending);
    if (before === toSort.toString()) {
        toSort.sort(descending);
    }

    rows.each((index1, element1) => {
        for (let nthChild = 1; nthChild < $(element1).children().length; nthChild++) {
            $($(element1).find("div")[nthChild]).html(toSort[nthChild - 1][index1]);
        }
    });
}

$(document).ready(() => {
    let table1 = $("#table01");
    let table2 = $("#table02");
    let rows1 = table1.find("tr");
    let rows2 = table2.find("tr");

    table1.find("th").each((index, element) => {
        let el = $(element);
        el.click(() => {
            sortElementsCols(index, rows1);
        });
    });

    table2.find("th").each((index, element) => {
        let el = $(element);
        el.click(() => {
            sortElementsLines(index, rows2);
        });
    });
});