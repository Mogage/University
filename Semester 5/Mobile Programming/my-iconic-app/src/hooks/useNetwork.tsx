import { useContext, useEffect, useState } from "react";
import { Network, ConnectionStatus } from "@capacitor/network";
import { PluginListenerHandle } from "@capacitor/core";
import { AuthContext } from "../auth";
import { Preferences } from "@capacitor/preferences";
import { syncCalendarItems } from "../todo/calendarItemApi";

const initialState = {
  connected: false,
  connectionType: "unknown",
};

export const useNetwork = () => {
  const [networkStatus, setNetworkStatus] = useState(initialState);
  const { token } = useContext(AuthContext);
  useEffect(() => {
    let handler: PluginListenerHandle;
    registerNetworkStatusChange();
    Network.getStatus().then(handleNetworkStatusChange);
    let canceled = false;
    return () => {
      canceled = true;
      handler?.remove();
    };

    async function registerNetworkStatusChange() {
      handler = await Network.addListener(
        "networkStatusChange",
        handleNetworkStatusChange
      );
    }

    function sendUpdatesToServer() {
      (async () => {
        if (!networkStatus.connected) return;
        const calendarItems = await Preferences.get({ key: "calendarItems" });
        if (calendarItems.value) {
          try {
            syncCalendarItems(token, JSON.parse(calendarItems.value));
          } catch (e) {
            console.log(e);
          }
        }
      })();
    }

    async function handleNetworkStatusChange(status: ConnectionStatus) {
      console.log("useNetwork - status change", status);
      if (!canceled) {
        setNetworkStatus(status);
      }
      sendUpdatesToServer();
    }
  }, [token]);
  return { networkStatus };
};
