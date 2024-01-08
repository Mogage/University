import Router from "koa-router";
import calendarItemStore from "./store";
import { broadcast } from "../utils";

export const router = new Router();

router.get("/", async (ctx) => {
  const response = ctx.response;
  const userId = ctx.state.user._id;
  response.body = await calendarItemStore.find({ userId });
  response.status = 200; // ok
});

router.get("/:id", async (ctx) => {
  const userId = ctx.state.user._id;
  const calendarItem = await calendarItemStore.findOne({ _id: ctx.params.id });
  const response = ctx.response;
  if (calendarItem) {
    if (calendarItem.userId === userId) {
      response.body = calendarItem;
      response.status = 200; // ok
    } else {
      response.status = 403; // forbidden
    }
  } else {
    response.status = 404; // not found
  }
});

const createCalendarItem = async (ctx, calendarItem, response) => {
  try {
    const userId = ctx.state.user._id;
    calendarItem.userId = userId;
    calendarItem._id = undefined;
    response.body = await calendarItemStore.insert(calendarItem);
    response.status = 201; // created
    calendarItem._id = response.body._id;
    broadcast(userId, { type: "created", payload: calendarItem });
  } catch (err) {
    response.body = { message: err.message };
    response.status = 400; // bad request
  }
};

router.put("/sync", async (ctx) => {
  try {
    const userId = ctx.state.user._id;
    console.log("userId: " + userId);
    const calendarItems = ctx.request.body;
    console.log("calendarItems=" + calendarItems);
    const response = ctx.response;
    const currentCalendarItems = await calendarItemStore.find({ userId });
    for (var calendarItem of calendarItems) {
      calendarItem.userId = userId;
      if (!calendarItem._id) {
        await createMovie(ctx, calendarItem, ctx.response);
      } else {
        const existingCalendarItem = currentCalendarItems.find(
          (m) => m._id === calendarItem._id
        );
        if (existingCalendarItem) {
          await calendarItemStore.update(
            { _id: calendarItem._id },
            calendarItem
          );
        }
      }
    }

    response.body = await calendarItemStore.find({ userId });
    response.status = 200;
    //broadcast(userId, { type: 'sync', payload: response.body });
  } catch (err) {
    console.log(err);
    ctx.response.body = { message: err.message };
    ctx.response.status = 400;
  }
});

router.post(
  "/",
  async (ctx) => await createCalendarItem(ctx, ctx.request.body, ctx.response)
);

router.put("/:id", async (ctx) => {
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

router.del("/:id", async (ctx) => {
  const userId = ctx.state.user._id;
  const calendarItem = await calendarItemStore.findOne({ _id: ctx.params.id });
  if (calendarItem && userId !== calendarItem.userId) {
    ctx.response.status = 403; // forbidden
  } else {
    await calendarItemStore.remove({ _id: ctx.params.id });
    ctx.response.status = 204; // no content
  }
});
