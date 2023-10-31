import { useCallback, useEffect, useReducer } from 'react';
import { getLogger } from '../core';
import { CalendarItemProps } from './CalendarItemProps';
import { getCalendarItems } from './calendarItemApi';

const log = getLogger('useItems');

export interface CalendarItemsState {
  calendarItems?: CalendarItemProps[],
  fetching: boolean,
  fetchingError?: Error,
}

export interface CalendarItemsProps extends CalendarItemsState {
  addCalendarItem: () => void,
}

interface ActionProps {
  type: string,
  payload?: any,
}

const initialState: CalendarItemsState = {
  calendarItems: undefined,
  fetching: false,
  fetchingError: undefined,
};

const FETCH_CALENDAR_ITEMS_STARTED = 'FETCH_CALENDAR_ITEMS_STARTED';
const FETCH_CALENDAR_ITEMS_SUCCEEDED = 'FETCH_CALENDAR_ITEMS_SUCCEEDED';
const FETCH_CALENDAR_ITEMS_FAILED = 'FETCH_CALENDAR_ITEMS_FAILED';

const reducer: (state: CalendarItemsState, action: ActionProps) => CalendarItemsState =
  (state, { type, payload }) => {
    switch(type) {
      case FETCH_CALENDAR_ITEMS_STARTED:
        return { ...state, fetching: true };
      case FETCH_CALENDAR_ITEMS_SUCCEEDED:
        return { ...state, calendarItems: payload.calendarItems, fetching: false };
      case FETCH_CALENDAR_ITEMS_FAILED:
        return { ...state, fetchingError: payload.error, fetching: false };
      default:
        return state;
    }
  };

export const useCalendarItems: () => CalendarItemsProps = () => {
  const [state, dispatch] = useReducer(reducer, initialState);
  const { calendarItems, fetching, fetchingError } = state;
  const addCalendarItem = useCallback(() => {
    log('addCalendarItem - TODO');
  }, []);
  useEffect(getCalendarItemsEffect, [dispatch]);
  log(`returns - fetching = ${fetching}, calendar items = ${JSON.stringify(calendarItems)}`);
  return {
    calendarItems,
    fetching,
    fetchingError,
    addCalendarItem,
  };

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
};
