function sortTable(lineNumber) {
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
        if (a[lineNumber] < b[lineNumber]) {
            return -1;
        }
        if (a[lineNumber] > b[lineNumber]) {
            return 1;
        }
        return 0;
    }

    function descending(a, b) {
        if (a[lineNumber] > b[lineNumber]) {
            return -1;
        }
        if (a[lineNumber] < b[lineNumber]) {
            return 1;
        }
        return 0;
    }

    let before = columns.toString();
    if (!isNaN(parseInt(columns[0][lineNumber]))) {
        for (let index = 0; index < columns.length; index++) {
            columns[index][lineNumber] = parseInt(columns[index][lineNumber]);
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