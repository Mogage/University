
$(document).ready(function () {
    $("#firstSelect").dblclick(function () {
        $("#secondSelect").append($(":selected"))
    })
    $("#secondSelect").dblclick(function () {
        $("#firstSelect").append($(":selected"))
    })
});
