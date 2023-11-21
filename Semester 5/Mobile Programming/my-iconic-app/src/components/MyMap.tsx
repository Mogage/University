import { GoogleMap } from "@capacitor/google-maps";
import { useEffect, useRef, useState } from "react";
import { mapsApiKey } from "./mapsApiKey";
import { IonButton } from "@ionic/react";

interface MyMapProps {
  lat: number;
  lng: number;
  onMapClick: (e: any) => void;
  onMarkerClick: (e: any) => void;
}

const MyMap: React.FC<MyMapProps> = ({
  lat,
  lng,
  onMapClick,
  onMarkerClick,
}) => {
  const mapRef = useRef<HTMLElement>(null);
  const [buttonText, setButtonText] = useState("Open Map");
  const [viewMap, setViewMap] = useState(false);
  let myLocationMarkerId: string;
  useEffect(myMapEffect, [mapRef.current]);

  return (
    <div className="component-wrapper">
      <IonButton
        onClick={async () => {
          if (buttonText === "Close Map") {
            setViewMap(false);
            setButtonText("Open Map");
            return;
          }
          setViewMap(true);
          setButtonText("Close Map");
        }}
      >
        {buttonText}
      </IonButton>

      {viewMap && (
        <capacitor-google-map
          ref={mapRef}
          style={{
            display: "block",
            width: 400,
            height: 400,
          }}
        ></capacitor-google-map>
      )}
    </div>
  );

  function myMapEffect() {
    let canceled = false;
    let googleMap: GoogleMap | null = null;
    createMap();
    return () => {
      canceled = true;
      googleMap?.removeAllMapListeners();
    };

    async function addMarker({
      latitude,
      longitude,
    }: {
      latitude: number;
      longitude: number;
    }) {
      if (!googleMap) {
        return;
      }
      const coordinate = {
        lat: latitude,
        lng: longitude,
      };
      googleMap.removeMarker(myLocationMarkerId);
      myLocationMarkerId = await googleMap.addMarker({
        coordinate,
        title: "My location",
      });
    }

    async function createMap() {
      if (!mapRef.current) {
        return;
      }
      googleMap = await GoogleMap.create({
        id: "my-cool-map",
        element: mapRef.current,
        apiKey: mapsApiKey,
        config: {
          center: { lat, lng },
          zoom: 11,
        },
      });
      console.log("gm created");
      myLocationMarkerId = await googleMap.addMarker({
        coordinate: { lat, lng },
        title: "My location",
      });
      await googleMap.setOnMapClickListener(({ latitude, longitude }) => {
        addMarker({ latitude, longitude });
        onMapClick({ latitude, longitude });
      });
      await googleMap.setOnMarkerClickListener(
        ({ markerId, latitude, longitude }) => {
          onMarkerClick({ markerId, latitude, longitude });
        }
      );
    }
  }
};

export default MyMap;
