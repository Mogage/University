import React, { useCallback, useContext, useEffect, useReducer } from "react";
import PropTypes from "prop-types";
import { getLogger } from "../core";
import { CalendarItemProps } from "./CalendarItemProps";
import {
  createCalendarItem,
  getCalendarItems,
  newWebSocket,
  updateCalendarItem,
} from "./calendarItemApi";
import { AuthContext } from "../auth";
import { Preferences } from "@capacitor/preferences";
import { NetworkStatusContext } from "../hooks/NetworkStatusProvider";

const log = getLogger("CalendarItemProvider");

type SaveCalendarItemFn = (calendarItem: CalendarItemProps) => Promise<any>;

export interface CalendarItemsState {
  calendarItems?: CalendarItemProps[];
  fetching: boolean;
  fetchingError?: Error | null;
  saving: boolean;
  savingError?: Error | null;
  saveCalendarItem?: SaveCalendarItemFn;
  saveCalendarItemOffline?: SaveCalendarItemFn;
}

interface ActionProps {
  type: string;
  payload?: any;
}

const initialState: CalendarItemsState = {
  fetching: false,
  saving: false,
};

const FETCH_CALENDAR_ITEMS_STARTED = "FETCH_CALENDAR_ITEMS_STARTED";
const FETCH_CALENDAR_ITEMS_SUCCEEDED = "FETCH_CALENDAR_ITEMS_SUCCEEDED";
const FETCH_CALENDAR_ITEMS_FAILED = "FETCH_CALENDAR_ITEMS_FAILED";
const SAVE_CALENDAR_ITEM_STARTED = "SAVE_CALENDAR_ITEM_STARTED";
const SAVE_CALENDAR_ITEM_SUCCEEDED = "SAVE_CALENDAR_ITEM_SUCCEEDED";
const SAVE_CALENDAR_ITEM_FAILED = "SAVE_CALENDAR_ITEM_FAILED";

const reducer: (
  state: CalendarItemsState,
  action: ActionProps
) => CalendarItemsState = (state, { type, payload }) => {
  switch (type) {
    case FETCH_CALENDAR_ITEMS_STARTED:
      return { ...state, fetching: true, fetchingError: null };
    case FETCH_CALENDAR_ITEMS_SUCCEEDED:
      return {
        ...state,
        calendarItems: payload.calendarItems,
        fetching: false,
      };
    case FETCH_CALENDAR_ITEMS_FAILED:
      return { ...state, fetchingError: payload.error, fetching: false };
    case SAVE_CALENDAR_ITEM_STARTED:
      return { ...state, savingError: null, saving: true };
    case SAVE_CALENDAR_ITEM_SUCCEEDED:
      const calendarItems = [...(state.calendarItems || [])];
      const calendarItem = payload.calendarItem;
      const index = calendarItems.findIndex(
        (it) => it._id === calendarItem._id
      );
      if (index === -1) {
        calendarItems.splice(0, 0, calendarItem);
      } else {
        calendarItems[index] = calendarItem;
      }
      return { ...state, calendarItems, saving: false };
    case SAVE_CALENDAR_ITEM_FAILED:
      return { ...state, savingError: payload.error, saving: false };
    default:
      return state;
  }
};

export const CalendarItemContext =
  React.createContext<CalendarItemsState>(initialState);

interface CalendarItemProviderProps {
  children: PropTypes.ReactNodeLike;
}

export const CalendarItemProvider: React.FC<CalendarItemProviderProps> = ({
  children,
}) => {
  const { token } = useContext(AuthContext);
  const [state, dispatch] = useReducer(reducer, initialState);
  const { calendarItems, fetching, fetchingError, saving, savingError } = state;
  useEffect(getCalendarItemsEffect, [token]);
  useEffect(wsEffect, [token]);
  const saveCalendarItem = useCallback<SaveCalendarItemFn>(
    saveCalendarItemCallback,
    [token]
  );
  const saveCalendarItemOffline = useCallback<SaveCalendarItemFn>(
    saveCalendarItemOfflineCallback,
    []
  );
  const networkStatus = useContext(NetworkStatusContext);
  const value = {
    calendarItems,
    fetching,
    fetchingError,
    saving,
    savingError,
    saveCalendarItem,
    saveCalendarItemOffline,
    networkStatus,
  };
  log("returns");
  return (
    <CalendarItemContext.Provider value={value}>
      {children}
    </CalendarItemContext.Provider>
  );

  function getCalendarItemsEffect() {
    let canceled = false;
    if (token) {
      fetchCalendarItems();
    }
    return () => {
      canceled = true;
    };

    async function fetchCalendarItems() {
      try {
        if (networkStatus.connected === false) {
          log("No network connection - fetchItems failed");
          return;
        }
        log("fetchCalendarItems started");
        dispatch({ type: FETCH_CALENDAR_ITEMS_STARTED });
        const calendarItems = await getCalendarItems(token);
        log("fetchCalendarItems succeeded");
        if (!canceled) {
          dispatch({
            type: FETCH_CALENDAR_ITEMS_SUCCEEDED,
            payload: { calendarItems },
          });
        }
      } catch (error) {
        log("fetchCalendarItems failed");
        if (!canceled) {
          dispatch({ type: FETCH_CALENDAR_ITEMS_FAILED, payload: { error } });
        }
      }
    }
  }

  async function saveCalendarItemCallback(calendarItem: CalendarItemProps) {
    try {
      log("saveCalendarItem started");
      dispatch({ type: SAVE_CALENDAR_ITEM_STARTED });
      let savedCalendarItem;
      let calendarItems = await Preferences.get({ key: "calendarItems" });

      if (!calendarItems.value) {
        return;
      }
      let calendarItemsArray = JSON.parse(calendarItems.value);
      if (calendarItem._id) {
        savedCalendarItem = await updateCalendarItem(token, calendarItem);
        const index = calendarItemsArray.findIndex(
          (it: CalendarItemProps) => it._id === calendarItem._id
        );
        if (index === -1) {
          calendarItemsArray.splice(0, 0, savedCalendarItem);
        } else {
          calendarItemsArray[index] = savedCalendarItem;
        }
      } else {
        savedCalendarItem = await createCalendarItem(token, calendarItem);
        calendarItemsArray.splice(0, 0, savedCalendarItem);
      }
      await Preferences.set({
        key: "calendarItems",
        value: JSON.stringify(calendarItemsArray),
      });
      log("saveCalendarItem succeeded");
      dispatch({
        type: SAVE_CALENDAR_ITEM_SUCCEEDED,
        payload: { calendarItem: savedCalendarItem },
      });
    } catch (error) {
      log("saveCalendarItem failed");
      dispatch({ type: SAVE_CALENDAR_ITEM_FAILED, payload: { error } });
    }
  }

  async function saveCalendarItemOfflineCallback(
    calendarItem: CalendarItemProps
  ) {
    try {
      log("saveCalendarItemOffline started");
      dispatch({ type: SAVE_CALENDAR_ITEM_STARTED });
      let savedCalendarItem;
      let calendarItems = await Preferences.get({ key: "calendarItems" });

      if (calendarItems.value) {
        let calendarItemsArray = JSON.parse(calendarItems.value);
        const index = calendarItemsArray.findIndex(
          (it: CalendarItemProps) => it._id === calendarItem._id
        );
        if (index === -1) {
          calendarItemsArray.splice(0, 0, calendarItem);
        } else {
          calendarItemsArray[index] = calendarItem;
        }
        await Preferences.set({
          key: "calendarItems",
          value: JSON.stringify(calendarItemsArray),
        });
        savedCalendarItem = calendarItem;
      } else {
        await Preferences.set({
          key: "calendarItems",
          value: JSON.stringify([calendarItem]),
        });
        savedCalendarItem = calendarItem;
      }

      dispatch({
        type: SAVE_CALENDAR_ITEM_SUCCEEDED,
        payload: { calendarItem: savedCalendarItem },
      });
    } catch (error) {
      log("saveCalendarItemOffline failed");
      dispatch({ type: SAVE_CALENDAR_ITEM_FAILED, payload: { error } });
    }
  }

  function wsEffect() {
    let canceled = false;
    log("wsEffect - connecting");
    let closeWebSocket: () => void;
    if (token?.trim()) {
      closeWebSocket = newWebSocket(token, (message) => {
        if (canceled) {
          return;
        }
        const { type, payload: calendarItem } = message;
        log(`ws message, calendarItem ${type}`);
        if (type === "created" || type === "updated") {
          dispatch({
            type: SAVE_CALENDAR_ITEM_SUCCEEDED,
            payload: { calendarItem },
          });
        } else if (type === "synced") {
          const calendarItems = JSON.stringify(calendarItem);
          log(`ws message, items ${calendarItems}`);
          dispatch({ type: FETCH_CALENDAR_ITEMS_STARTED });
          getCalendarItemsEffect();
        }
      });
      return () => {
        log("wsEffect - disconnecting");
        canceled = true;
        closeWebSocket();
      };
    }
  }
};
