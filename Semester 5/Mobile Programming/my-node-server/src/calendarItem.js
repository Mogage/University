import Router from "koa-router";
import dataStore from "nedb-promise";
import { broadcast } from "./wss.js";

export class CalendarItemStore {
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
    if (!calendarItem.title) {
      throw new Error("Missing text property");
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

const calendarItemStore = new CalendarItemStore({
  filename: "./db/calendarItems.json",
  autoload: true,
});

export const calendarItemRouter = new Router();

calendarItemRouter.get("/", async (ctx) => {
  const userId = ctx.state.user._id;
  ctx.response.body = await calendarItemStore.find({ userId });
  ctx.response.status = 200; // ok
});

calendarItemRouter.get("/:id", async (ctx) => {
  const userId = ctx.state.user._id;
  const calendarItem = await noteStore.findOne({ _id: ctx.params.id });
  const response = ctx.response;
  if (calendarItem) {
    if (calendarItem.userId === userId) {
      ctx.response.body = calendarItem;
      ctx.response.status = 200; // ok
    } else {
      ctx.response.status = 403; // forbidden
    }
  } else {
    ctx.response.status = 404; // not found
  }
});

calendarItemRouter.put("/sync", async (ctx) => {
  try {
    const userId = ctx.state.user._id;
    const calendarItems = ctx.request.body;
    const response = ctx.response;
    const calendarItemsFromDb = await calendarItemStore.find({ userId });
    for (let calendarItem of calendarItems) {
      if (!calendarItem._id) {
        calendarItem.userId = userId;
        await calendarItemStore.insert(calendarItem);
      } else {
        await calendarItemStore.update({ _id: calendarItem._id }, calendarItem);
      }
    }
    const calendarItemsToDelete = calendarItemsFromDb.filter(
      (calendarItemFromDb) =>
        !calendarItems.find(
          (calendarItem) => calendarItem._id === calendarItemFromDb._id
        )
    );
    await Promise.all(
      calendarItemsToDelete.map((calendarItem) =>
        calendarItemStore.remove({ _id: calendarItem._id })
      )
    );
    response.body = await calendarItemStore.find({ userId });
    response.status = 200;
    broadcast(userId, { type: "synced", payload: response.body });
  } catch (err) {
    ctx.response.body = { message: err.message };
    ctx.response.status = 400;
  }
});

const createCalendarItem = async (ctx, calendarItem, response) => {
  try {
    const userId = ctx.state.user._id;
    calendarItem.userId = userId;
    response.body = await calendarItemStore.insert(calendarItem);
    response.status = 201; // created
    broadcast(userId, { type: "created", payload: response.body });
  } catch (err) {
    response.body = { message: err.message };
    response.status = 400; // bad request
  }
};

calendarItemRouter.post(
  "/",
  async (ctx) => await createCalendarItem(ctx, ctx.request.body, ctx.response)
);

calendarItemRouter.put("/:id", async (ctx) => {
  const calendarItem = ctx.request.body;
  const id = ctx.params.id;
  const calendarItemId = calendarItem._id;
  const response = ctx.response;
  if (calendarItemId && calendarItemId !== id) {
    response.body = { message: "Param id and body _id should be the same" };
    response.status = 400; // bad request
    return;
  }
  if (!calendarItemId) {
    await createCalendarItem(ctx, calendarItem, response);
  } else {
    const userId = ctx.state.user._id;
    calendarItem.userId = userId;
    const updatedCount = await calendarItemStore.update(
      { _id: id },
      calendarItem
    );
    if (updatedCount === 1) {
      response.body = calendarItem;
      response.status = 200; // ok
      broadcast(userId, { type: "updated", payload: calendarItem });
    } else {
      response.body = { message: "Resource no longer exists" };
      response.status = 405; // method not allowed
    }
  }
});

calendarItemRouter.del("/:id", async (ctx) => {
  const userId = ctx.state.user._id;
  const calendarItem = await calendarItemStore.findOne({ _id: ctx.params.id });
  if (calendarItem && userId !== calendarItem.userId) {
    ctx.response.status = 403; // forbidden
  } else {
    await calendarItemStore.remove({ _id: ctx.params.id });
    ctx.response.status = 204; // no content
  }
});
