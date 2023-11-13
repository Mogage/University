import { useContext, useEffect } from "react";
import { Preferences } from "@capacitor/preferences";
import { AuthContext } from "../auth";

export const usePreferences = () => {
  const { token, isAuthenticated } = useContext(AuthContext);
  useEffect(() => {
    runPreferences(token);
  }, [token, isAuthenticated]);

  function runPreferences(token: string) {
    (async () => {
      await Preferences.set({
        key: "token",
        value: token,
      });
    })();
  }

  function runPreferencesDemo() {
    (async () => {
      // Saving ({ key: string, value: string }) => Promise<void>
      await Preferences.set({
        key: token,
        value: JSON.stringify({
          username: "nicu",
          password: "nicu",
        }),
      });

      // Loading value by key ({ key: string }) => Promise<{ value: string | null }>
      const res = await Preferences.get({ key: token });
      if (res.value) {
        console.log("User found", JSON.parse(res.value));
      } else {
        console.log("User not found");
      }

      // Loading keys () => Promise<{ keys: string[] }>
      const { keys } = await Preferences.keys();
      console.log("Keys found", keys);

      // Removing value by key, ({ key: string }) => Promise<void>
      await Preferences.remove({ key: token });
      console.log("Keys found after remove", await Preferences.keys());

      // Clear storage () => Promise<void>
      await Preferences.clear();
    })();
  }
};
