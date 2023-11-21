import { useCallback } from "react";
import {
  Camera,
  CameraResultType,
  CameraSource,
  Photo,
} from "@capacitor/camera";
import { defineCustomElements } from "@ionic/pwa-elements/loader";

export function useCamera() {
  defineCustomElements(window);
  const getPhoto = useCallback<() => Promise<Photo>>(
    () =>
      Camera.getPhoto({
        resultType: CameraResultType.Base64,
        source: CameraSource.Camera,
        quality: 100,
      }),
    []
  );

  return {
    getPhoto,
  };
}
