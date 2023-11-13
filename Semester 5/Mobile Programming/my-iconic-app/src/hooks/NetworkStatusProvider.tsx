import React, { createContext, useContext, useEffect } from "react";
import { useNetwork } from "./useNetwork";
// Define the NetworkStatus context.
export const NetworkStatusContext = createContext({
  connected: false,
  connectionType: "unknown",
});

const NetworkStatusProvider = ({ children }) => {
  const { networkStatus } = useNetwork();

  return (
    <NetworkStatusContext.Provider value={networkStatus}>
      {children}
    </NetworkStatusContext.Provider>
  );
};

export default NetworkStatusProvider;
