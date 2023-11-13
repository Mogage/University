import { useContext, useEffect } from "react";
import { Preferences } from "@capacitor/preferences";
import { AuthContext } from "../auth";
import { CalendarItemContext } from "../todo/CalendarItemProvider";

export const usePreferencesCalendarItems = () => {
  const { calendarItems, fetching, fetchingError } =
    useContext(CalendarItemContext);
  useEffect(() => {
    if (!fetching && calendarItems) {
      calendarItems.sort((a, b) => (a._id! > b._id! ? 1 : -1));
      runPreferences();
    }
  }, [calendarItems, fetching]);

  function runPreferences() {
    (async () => {
      await Preferences.remove({ key: "calendarItems" });
      await Preferences.set({
        key: "calendarItems",
        value: JSON.stringify(calendarItems),
      });
    })();
  }
};
