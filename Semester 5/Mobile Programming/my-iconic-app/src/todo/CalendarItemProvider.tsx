import React, { useCallback, useEffect, useReducer } from 'react';
import PropTypes from 'prop-types';
import { getLogger } from '../core';
import { CalendarItemProps } from './CalendarItemProps';
import { createCalendarItem, getCalendarItems, newWebSocket, updateCalendarItem } from './calendarItemApi';

const log = getLogger('CalendarItemProvider');

type SaveCalendarItemFn = (calendarItem: CalendarItemProps) => Promise<any>;

export interface CalendarItemsState {
  calendarItems?: CalendarItemProps[],
  fetching: boolean,
  fetchingError?: Error | null,
  saving: boolean,
  savingError?: Error | null,
  saveCalendarItem?: SaveCalendarItemFn,
}

interface ActionProps {
  type: string,
  payload?: any,
}

const initialState: CalendarItemsState = {
  fetching: false,
  saving: false,
};

const FETCH_CALENDAR_ITEMS_STARTED = 'FETCH_CALENDAR_ITEMS_STARTED';
const FETCH_CALENDAR_ITEMS_SUCCEEDED = 'FETCH_CALENDAR_ITEMS_SUCCEEDED';
const FETCH_CALENDAR_ITEMS_FAILED = 'FETCH_CALENDAR_ITEMS_FAILED';
const SAVE_CALENDAR_ITEM_STARTED = 'SAVE_CALENDAR_ITEM_STARTED';
const SAVE_CALENDAR_ITEM_SUCCEEDED = 'SAVE_CALENDAR_ITEM_SUCCEEDED';
const SAVE_CALENDAR_ITEM_FAILED = 'SAVE_CALENDAR_ITEM_FAILED';

const reducer: (state: CalendarItemsState, action: ActionProps) => CalendarItemsState =
  (state, { type, payload }) => {
    switch(type) {
      case FETCH_CALENDAR_ITEMS_STARTED:
        return { ...state, fetching: true, fetchingError: null };
      case FETCH_CALENDAR_ITEMS_SUCCEEDED:
        return { ...state, calendarItems: payload.calendarItems, fetching: false };
      case FETCH_CALENDAR_ITEMS_FAILED:
        return { ...state, fetchingError: payload.error, fetching: false };
      case SAVE_CALENDAR_ITEM_STARTED:
        return { ...state, savingError: null, saving: true };
      case SAVE_CALENDAR_ITEM_SUCCEEDED:
        const calendarItems = [...(state.calendarItems || [])];
        const calendarItem = payload.calendarItem;
        const index = calendarItems.findIndex(it => it.id === calendarItem.id);
        if (index === -1) {
          calendarItems.splice(0, 0, calendarItem);
        } else {
          calendarItems[index] = calendarItem;
        }
        return { ...state,  calendarItems, saving: false };
      case SAVE_CALENDAR_ITEM_FAILED:
        return { ...state, savingError: payload.error, saving: false };
      default:
        return state;
    }
  };

export const CalendarItemContext = React.createContext<CalendarItemsState>(initialState);

interface CalendarItemProviderProps {
  children: PropTypes.ReactNodeLike,
}

export const CalendarItemProvider: React.FC<CalendarItemProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(reducer, initialState);
  const { calendarItems, fetching, fetchingError, saving, savingError } = state;
  useEffect(getCalendarItemsEffect, []);
  useEffect(wsEffect, []);
  const saveCalendarItem = useCallback<SaveCalendarItemFn>(saveCalendarItemCallback, []);
  const value = { calendarItems, fetching, fetchingError, saving, savingError, saveCalendarItem };
  log('returns');
  return (
    <CalendarItemContext.Provider value={value}>
      {children}
    </CalendarItemContext.Provider>
  );

  function getCalendarItemsEffect() {
    let canceled = false;
    fetchCalendarItems();
    return () => {
      canceled = true;
    }

    async function fetchCalendarItems() {
      try {
        log('fetchCalendarItems started');
        dispatch({ type: FETCH_CALENDAR_ITEMS_STARTED });
        const calendarItems = await getCalendarItems();
        log('fetchCalendarItems succeeded');
        if (!canceled) {
          dispatch({ type: FETCH_CALENDAR_ITEMS_SUCCEEDED, payload: { calendarItems } });
        }
      } catch (error) {
        log('fetchCalendarItems failed');
        if (!canceled) {
          dispatch({ type: FETCH_CALENDAR_ITEMS_FAILED, payload: { error } });
        }
      }
    }
  }

  async function saveCalendarItemCallback(calendarItem: CalendarItemProps) {
    try {
      log('saveCalendarItem started');
      dispatch({ type: SAVE_CALENDAR_ITEM_STARTED });
      const savedCalendarItem = await (calendarItem.id ? updateCalendarItem(calendarItem) : createCalendarItem(calendarItem));
      log('saveCalendarItem succeeded');
      dispatch({ type: SAVE_CALENDAR_ITEM_SUCCEEDED, payload: { calendarItem: savedCalendarItem } });
    } catch (error) {
      log('saveCalendarItem failed');
      dispatch({ type: SAVE_CALENDAR_ITEM_FAILED, payload: { error } });
    }
  }

  function wsEffect() {
    let canceled = false;
    log('wsEffect - connecting');
    const closeWebSocket = newWebSocket(message => {
      if (canceled) {
        return;
      }
      const { event, payload: { calendarItem }} = message;
      log(`ws message, calendarItem ${event}`);
      if (event === 'created' || event === 'updated') {
        dispatch({ type: SAVE_CALENDAR_ITEM_SUCCEEDED, payload: { calendarItem } });
      }
    });
    return () => {
      log('wsEffect - disconnecting');
      canceled = true;
      closeWebSocket();
    }
  }
};
