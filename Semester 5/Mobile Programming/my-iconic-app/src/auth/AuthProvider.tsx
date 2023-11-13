import React, { useCallback, useEffect, useState } from "react";
import PropTypes from "prop-types";
import { getLogger } from "../core";
import { login as loginApi } from "./authAPI";
import { usePreferences } from "../hooks/usePreferencesToken";
import { Preferences } from "@capacitor/preferences";

const log = getLogger("AuthProvider");

type LoginFn = (username?: string, password?: string) => void;
type LogoutFn = (token?: string) => void;

export interface AuthState {
  authenticationError: Error | null;
  isAuthenticated: boolean;
  isAuthenticating: boolean;
  login?: LoginFn;
  logout?: LogoutFn;
  pendingAuthentication?: boolean;
  username?: string;
  password?: string;
  token: string;
}

const initialState: AuthState = {
  isAuthenticated: false,
  isAuthenticating: false,
  authenticationError: null,
  pendingAuthentication: false,
  token: "",
};

export const AuthContext = React.createContext<AuthState>(initialState);

interface AuthProviderProps {
  children: PropTypes.ReactNodeLike;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  useEffect(() => {
    const fetchToken = async () => {
      const token = await Preferences.get({ key: "token" });
      if (token?.value) {
        setState({
          ...state,
          token: token.value,
          isAuthenticated: true,
        });
      }
    };
    fetchToken();
  }, []);

  const [state, setState] = useState<AuthState>(initialState);
  const {
    isAuthenticated,
    isAuthenticating,
    authenticationError,
    pendingAuthentication,
    token,
  } = state;
  const login = useCallback<LoginFn>(loginCallback, []);
  const logout = useCallback<LogoutFn>(logoutCallback, []);
  useEffect(authenticationEffect, [pendingAuthentication]);
  const value = {
    isAuthenticated,
    login,
    logout,
    isAuthenticating,
    authenticationError,
    token,
  };
  log("render");
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;

  function logoutCallback(token?: string) {
    const unsetToken = async () => {
      await Preferences.remove({ key: "token" });
      setState({
        ...state,
        token: "",
        isAuthenticated: false,
      });
    };
    unsetToken();
  }

  function loginCallback(username?: string, password?: string): void {
    log("login");
    const fetchToken = async () => {
      const token = await Preferences.get({ key: "token" });
      if (token?.value) {
        // if token exists then we are authenticated
        setState({
          ...state,
          token: token.value,
          isAuthenticated: true,
        });
      } else {
        // we start the authentication chain
        setState({
          ...state,
          pendingAuthentication: true,
          username,
          password,
        });
      }
    };
    fetchToken();
  }

  function authenticationEffect() {
    let canceled = false;
    authenticate();
    return () => {
      canceled = true;
    };

    async function authenticate() {
      if (!pendingAuthentication) {
        log("authenticate, !pendingAuthentication, return");
        return;
      }
      try {
        log("authenticate...");
        setState({
          ...state,
          isAuthenticating: true,
        });
        const { username, password } = state;
        const { token } = await loginApi(username, password);
        if (canceled) {
          return;
        }
        log("authenticate succeeded");
        setState({
          ...state,
          token,
          pendingAuthentication: false,
          isAuthenticated: true,
          isAuthenticating: false,
        });
      } catch (error) {
        if (canceled) {
          return;
        }
        log("authenticate failed");
        setState({
          ...state,
          authenticationError: error as Error,
          pendingAuthentication: false,
          isAuthenticating: false,
        });
      }
    }
  }
};
