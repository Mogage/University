import dataStore from "nedb-promise";

class calendarItemStore {
  constructor({ filename, autoload }) {
    this.store = dataStore({ filename, autoload });
  }

  async find(props) {
    return this.store.find(props);
  }

  async findOne(props) {
    return this.store.findOne(props);
  }

  async insert(calendarItem) {
    let calendarItemTitle = calendarItem.title;
    if (!calendarItemTitle) {
      throw new Error("Missing title property");
    }
    return this.store.insert(calendarItem);
  }

  async update(props, calendarItem) {
    return this.store.update(props, calendarItem);
  }

  async remove(props) {
    return this.store.remove(props);
  }
}

export default new calendarItemStore({
  filename: "./db/calendarItems.json",
  autoload: true,
});
