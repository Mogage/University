const Koa = require('koa');
const app = new Koa();
const server = require('http').createServer(app.callback());
const WebSocket = require('ws');
const wss = new WebSocket.Server({ server });
const Router = require('koa-router');
const cors = require('koa-cors');
const bodyparser = require('koa-bodyparser');

app.use(bodyparser());
app.use(cors());
app.use(async (ctx, next) => {
  const start = new Date();
  await next();
  const ms = new Date() - start;
  console.log(`${ctx.method} ${ctx.url} ${ctx.response.status} - ${ms}ms`);
});

app.use(async (ctx, next) => {
  await new Promise(resolve => setTimeout(resolve, 2000));
  await next();
});

app.use(async (ctx, next) => {
  try {
    await next();
  } catch (err) {
    ctx.response.body = { issue: [{ error: err.message || 'Unexpected error' }] };
    ctx.response.status = 500;
  }
});

class CalendarItem {
  constructor({ id, title, type, noOfGuesst, startDate, endDate, isCompleted, doesRepeat }) {
    this.id = id;
    this.title = title;
    this.type = type;
    this.noOfGuesst = noOfGuesst;
    this.startDate = startDate;
    this.endDate = endDate;
    this.isCompleted = isCompleted;
    this.doesRepeat = doesRepeat;
  }
}

const calendarItems = [];
for (let i = 0; i < 5; i++) {
  calendarItems.push(new CalendarItem({ id: `${i}`, title: `item ${i}`, type: 'event', noOfGuesst: 0, startDate: new Date(Date.now() + i), endDate: new Date(Date.now() + i), isCompleted: false, doesRepeat: false }));
}
let lastUpdated = calendarItems[calendarItems.length - 1].date;
let lastId = calendarItems[calendarItems.length - 1].id;
const pageSize = 10;

const broadcast = data =>
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(data));
    }
  });

const router = new Router();

router.get('/calendarItem', ctx => {
  ctx.response.body = calendarItems;
  ctx.response.status = 200;
});

router.get('/calendarItem/:id', async (ctx) => {
  const itemId = ctx.request.params.id;
  const calendarItem = calendarItems.find(calendarItem => itemId === calendarItem.id);
  if (calendarItem) {
    ctx.response.body = calendarItem;
    ctx.response.status = 200; 
  } else {
    ctx.response.body = { message: `item with id ${itemId} not found` };
    ctx.response.status = 404; 
  }
});

const createItem = async (ctx) => {
  const calendarItem = ctx.request.body;
  if (!calendarItem.title || !calendarItem.type || !calendarItem.startDate || !calendarItem.endDate) { 
    ctx.response.body = { message: 'Invalid data' };
    ctx.response.status = 400;
    return;
  }
  calendarItem.id = `${parseInt(lastId) + 1}`;
  lastId = calendarItem.id;
  // item.startDate = new Date();
  calendarItem.version = 1;
  calendarItems.push(calendarItem);
  ctx.response.body = calendarItem;
  ctx.response.status = 201;
  broadcast({ event: 'created', payload: { calendarItem: calendarItem } });
};

router.post('/calendarItem', async (ctx) => {
  await createItem(ctx);
});

router.put('/calendarItem/:id', async (ctx) => {
  const id = ctx.params.id;
  const calendarItem = ctx.request.body;
  // item.date = new Date();
  const calendarItemId = calendarItem.id;
  if (calendarItemId && id !== calendarItem.id) {
    ctx.response.body = { message: `Param id and body id should be the same` };
    ctx.response.status = 400;
    return;
  }
  if (!calendarItemId) {
    await createItem(ctx);
    return;
  }
  const index = calendarItems.findIndex(calendarItem => calendarItem.id === id);
  if (index === -1) {
    ctx.response.body = { issue: [{ error: `item with id ${id} not found` }] };
    ctx.response.status = 400;
    return;
  }
  const itemVersion = parseInt(ctx.request.get('ETag')) || calendarItem.version;
  if (itemVersion < calendarItems[index].version) {
    ctx.response.body = { issue: [{ error: `Version conflict` }] };
    ctx.response.status = 409;
    return;
  }
  calendarItem.version++;
  calendarItems[index] = calendarItem;
  lastUpdated = new Date();
  ctx.response.body = calendarItem;
  ctx.response.status = 200; 
  broadcast({ event: 'updated', payload: { calendarItem: calendarItem } });
});

router.del('/calendarItem/:id', ctx => {
  const id = ctx.params.id;
  const index = calendarItems.findIndex(item => id === item.id);
  if (index !== -1) {
    const calendarItem = calendarItems[index];
    calendarItems.splice(index, 1);
    lastUpdated = new Date();
    broadcast({ event: 'deleted', payload: { calendarItem } });
  }
  ctx.response.status = 204; // no content
});

// setInterval(() => {
//   lastUpdated = new Date();
//   lastId = `${parseInt(lastId) + 1}`;
//   const item = new CalendarItem({ id: lastId, title: `item ${lastId}`, type: 'event', noOfGuesst: 0, startDate: new Date(Date.now() + lastId), endDate: new Date(Date.now() + lastId), isCompleted: false, doesRepeat: false });
//   items.push(item);
//   console.log(`New item: ${item.title}`);
//   broadcast({ event: 'created', payload: { item } });
// }, 5000);

app.use(router.routes());
app.use(router.allowedMethods());

server.listen(3000, () => {
  console.log('Server started on port 3000');
});

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');

  ws.on('message', (message) => {
    console.log(`Received: ${message}`);
  });

  ws.on('close', () => {
    console.log('WebSocket client disconnected');
  });
});