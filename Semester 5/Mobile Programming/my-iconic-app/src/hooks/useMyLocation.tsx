import { useEffect, useState } from "react";
import { Geolocation, Position } from "@capacitor/geolocation";

interface MyLocation {
  position?: Position | null;
  error?: Error;
}

export const useMyLocation = () => {
  const [state, setState] = useState<MyLocation>({});
  useEffect(() => {
    watchMyLocation();
  }, [setState]);
  return state;

  async function watchMyLocation() {
    let cancelled = false;
    let callbackId: string;
    try {
      const position = await Geolocation.getCurrentPosition();
      updateMyPosition("current", position);
    } catch (error) {
      updateMyPosition("current", null, error);
    }
    callbackId = await Geolocation.watchPosition({}, (position, error) => {
      updateMyPosition("watch", position, error);
    });
    return () => {
      cancelled = true;
      Geolocation.clearWatch({ id: callbackId });
    };

    function updateMyPosition(
      source: string,
      position?: Position | null,
      error: any = undefined
    ) {
      if (!cancelled && (position || error)) {
        setState({ position, error });
      }
    }
  }
};
