function moveOption(id) {
    let selectedList = document.getElementById(id);
    let value = selectedList.value;
    let index = selectedList.selectedIndex;
    let otherList = (id === "firstSelect") ? document.getElementById("secondSelect") : document.getElementById("firstSelect");
    let element = document.createElement("option");
    element.value = value;
    element.text = value;
    otherList.add(element);
    selectedList.remove(index);
    console.log("sal");
}