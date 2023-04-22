function sortTableColumns(colNumber) {
    let table = document.getElementById("table01");
    let rows = table.rows;
    let columns = [];
    for (let index = 0; index < rows[0].children.length - 1; index++) {
        columns.push([]);
    }
    for (let index = 0; index < rows.length; index++) {
        for (let nthChild = 1; nthChild < rows[index].children.length; nthChild++) {
            columns[nthChild - 1].push(rows[index].children[nthChild].querySelector("div").innerHTML);
        }
    }

    function ascending(a, b) {
        if (a[colNumber] < b[colNumber]) {
            return -1;
        }
        if (a[colNumber] > b[colNumber]) {
            return 1;
        }
        return 0;
    }

    function descending(a, b) {
        if (a[colNumber] > b[colNumber]) {
            return -1;
        }
        if (a[colNumber] < b[colNumber]) {
            return 1;
        }
        return 0;
    }

    let before = columns.toString();
    if (!isNaN(parseInt(columns[0][colNumber]))) {
        for (let index = 0; index < columns.length; index++) {
            columns[index][colNumber] = parseInt(columns[index][colNumber]);
        }
    }
    columns.sort(ascending);
    if (before === columns.toString()) {
        columns.sort(descending);
    }

    for (let index = 0; index < rows.length; index++) {
        for (let nthChild = 0; nthChild < rows[index].children.length - 1; nthChild++) {
            rows[index].children[nthChild + 1].querySelector("div").innerHTML = columns[nthChild][index];
        }
    }
}

function sortTableRows(lineNumber) {
    let table = document.getElementById("table02");
    let rows = table.rows;

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

    let toSort = [];
    for (let index = 1; index < rows.length; index++) {
        let toAdd = [rows[index].innerHTML];
        toAdd.sortValue = rows[index].children[lineNumber].querySelector("div").innerHTML;
        toSort.push(toAdd);
    }

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
        rows[index + 1].innerHTML = toSort[index];
    }
}