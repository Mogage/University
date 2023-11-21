import React, { memo, useEffect, useState } from "react";
import { IonImg, IonItem, IonLabel } from "@ionic/react";
import { getLogger } from "../core";
import { CalendarItemProps } from "./CalendarItemProps";
import { MyPhoto, usePhotos } from "../hooks/usePhotos";
import { useFilesystem } from "../hooks/useFilesystem";
import { usePreferences } from "../hooks/usePreferences";

interface CalendarItemPropsExt extends CalendarItemProps {
  onEdit: (id?: string) => void;
}

const formatDate = (date: Date): string => {
  const day = date.getDate().toString().padStart(2, "0");
  const month = (date.getMonth() + 1).toString().padStart(2, "0");
  const year = date.getFullYear();
  return `${day}/${month}/${year}`;
};

const CalendarItem: React.FC<CalendarItemPropsExt> = ({
  _id,
  title,
  type,
  noOfGuests,
  startDate,
  endDate,
  isCompleted,
  doesRepeat,
  latitude,
  longitude,
  onEdit,
}) => {
  const { readFile } = useFilesystem();
  const { get } = usePreferences();
  const [photos, setPhotos] = useState<MyPhoto[]>([]);
  useEffect(loadPhotos, [get, readFile, setPhotos]);

  function loadPhotos() {
    loadSavedPhotos();
    async function loadSavedPhotos() {
      const savedPhotoString = await get("photos" + _id);
      const savedPhotos = (
        savedPhotoString ? JSON.parse(savedPhotoString) : []
      ) as MyPhoto[];
      console.log("load", savedPhotos);
      for (let photo of savedPhotos) {
        const data = await readFile(photo.filepath);
        photo.webviewPath = `data:image/jpeg;base64,${data}`;
      }
      setPhotos(savedPhotos);
    }
  }

  return (
    <>
      {photos.length > 0 && (
        <IonImg
          src={photos[0]?.webviewPath || ""}
          style={{
            float: "left",
            width: "300px",
            height: "150px",
            objectFit: "cover",
            marginRight: "-300px",
          }}
        />
      )}
      <IonLabel style={{ textAlign: "center" }} onClick={() => onEdit(_id)}>
        <div>Title: {title}</div>
        <div>Type: {type}</div>
        <div>NoOfGuests: {noOfGuests}</div>
        <div>StartDate: {formatDate(new Date(startDate))}</div>
        <div>EndDate: {formatDate(new Date(endDate))}</div>
        <div>IsCompleted: {isCompleted ? "Yes" : "No"}</div>
        <div>DoesRepeat: {doesRepeat ? "Yes" : "No"}</div>
        <div>
          Latitude: {latitude} - Longitude: {longitude}
        </div>
        <br />
      </IonLabel>
    </>
  );
};

export default memo(CalendarItem);
