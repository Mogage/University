import axios from 'axios';
import { getLogger } from '../core';
import { CalendarItemProps } from './CalendarItemProps';

const log = getLogger('itemApi');

const baseUrl = 'localhost:3000';
const calendarItemUrl = `http://${baseUrl}/calendarItem`;

interface ResponseProps<T> {
  data: T;
}

function withLogs<T>(promise: Promise<ResponseProps<T>>, fnName: string): Promise<T> {
  log(`${fnName} - started`);
  return promise
    .then(res => {
      log(`${fnName} - succeeded`);
      return Promise.resolve(res.data);
    })
    .catch(err => {
      log(`${fnName} - failed`);
      return Promise.reject(err);
    });
}

const config = {
  headers: {
    'Content-Type': 'application/json'
  }
};

export const getCalendarItems: () => Promise<CalendarItemProps[]> = () => {
  return withLogs(axios.get(calendarItemUrl, config), 'getCalendarItems');
}

export const createCalendarItem: (calendarItem: CalendarItemProps) => Promise<CalendarItemProps[]> = calendarItem => {
  return withLogs(axios.post(calendarItemUrl, calendarItem, config), 'createCalendarItem');
}

export const updateCalendarItem: (calendarItem: CalendarItemProps) => Promise<CalendarItemProps[]> = calendarItem => {
  return withLogs(axios.put(`${calendarItemUrl}/${calendarItem.id}`, calendarItem, config), 'updateCalendarItem');
}

interface MessageData {
  event: string;
  payload: {
    calendarItem: CalendarItemProps;
  };
}

export const newWebSocket = (onMessage: (data: MessageData) => void) => {
  const ws = new WebSocket(`ws://${baseUrl}`)
  ws.onopen = () => {
    log('web socket onopen');
  };
  ws.onclose = () => {
    log('web socket onclose');
  };
  ws.onerror = error => {
    log('web socket onerror', error);
  };
  ws.onmessage = messageEvent => {
    log('web socket onmessage');
    onMessage(JSON.parse(messageEvent.data));
  };
  return () => {
    ws.close();
  }
}
