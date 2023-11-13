import axios from "axios";
import { authConfig, baseUrl, getLogger, withLogs } from "../core";
import { CalendarItemProps } from "./CalendarItemProps";

const log = getLogger("calendarItemApi");

const calendarItemUrl = `http://${baseUrl}/api/calendarItem`;

export const getCalendarItems: (
  token: string
) => Promise<CalendarItemProps[]> = (token) => {
  return withLogs(
    axios.get(calendarItemUrl, authConfig(token)),
    "getCalendarItems"
  );
};

export const createCalendarItem: (
  token: string,
  calendarItem: CalendarItemProps
) => Promise<CalendarItemProps[]> = (token, calendarItem) => {
  return withLogs(
    axios.post(calendarItemUrl, calendarItem, authConfig(token)),
    "createCalendarItem"
  );
};

export const updateCalendarItem: (
  token: string,
  calendarItem: CalendarItemProps
) => Promise<CalendarItemProps[]> = (token, calendarItem) => {
  return withLogs(
    axios.put(
      `${calendarItemUrl}/${calendarItem._id}`,
      calendarItem,
      authConfig(token)
    ),
    "updateCalendarItem"
  );
};

export const syncCalendarItems: (
  token: string,
  calendarItems: CalendarItemProps[]
) => Promise<CalendarItemProps[]> = (token, calendarItems) => {
  return withLogs(
    axios.put(`${calendarItemUrl}/sync`, calendarItems, authConfig(token)),
    "syncCalendarItems"
  );
};

interface MessageData {
  event: string;
  type: string;
  payload: {
    calendarItem: CalendarItemProps;
  };
}

export const newWebSocket = (
  token: string,
  onMessage: (data: MessageData) => void
) => {
  const ws = new WebSocket(`ws://${baseUrl}`);
  ws.onopen = () => {
    log("web socket onopen");
    ws.send(JSON.stringify({ type: "authorization", payload: { token } }));
  };
  ws.onclose = () => {
    log("web socket onclose");
  };
  ws.onerror = (error) => {
    log("web socket onerror", error);
  };
  ws.onmessage = (messageEvent) => {
    log("web socket onmessage");
    onMessage(JSON.parse(messageEvent.data));
  };
  return () => {
    ws.close();
  };
};
