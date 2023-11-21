import axios from "axios";
import { baseUrl, config, withLogs } from "../core";

const authUrl = `http://${baseUrl}/api/auth/`;

export interface AuthProps {
  token: string;
}

export const login: (
  username?: string,
  password?: string
) => Promise<AuthProps> = (username, password) => {
  return withLogs(
    axios.post(authUrl + "login", { username, password }, config),
    "login"
  );
};

export const signup: (
  username?: string,
  password?: string
) => Promise<AuthProps> = (username, password) => {
  return withLogs(
    axios.post(authUrl + "signup", { username, password }, config),
    "signup"
  );
};
