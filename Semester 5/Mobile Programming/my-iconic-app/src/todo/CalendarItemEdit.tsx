import React, { useCallback, useContext, useEffect, useState } from "react";
import {
  IonActionSheet,
  IonButton,
  IonButtons,
  IonCheckbox,
  IonCol,
  IonContent,
  IonDatetime,
  IonFabButton,
  IonGrid,
  IonHeader,
  IonIcon,
  IonImg,
  IonInput,
  IonLabel,
  IonLoading,
  IonPage,
  IonRow,
  IonSelect,
  IonSelectOption,
  IonTitle,
  IonToolbar,
} from "@ionic/react";
import { getLogger } from "../core";
import { CalendarItemContext } from "./CalendarItemProvider";
import { RouteComponentProps } from "react-router";
import { CalendarItemProps } from "./CalendarItemProps";
import { NetworkStatusContext } from "../hooks/NetworkStatusProvider";
import MyMap from "../components/MyMap";
import { useMyLocation } from "../hooks/useMyLocation";
import { camera, close, trash, save } from "ionicons/icons";
import { MyPhoto, usePhotos } from "../hooks/usePhotos";

const log = getLogger("ItemEdit");

interface CalendarItemEditProps
  extends RouteComponentProps<{
    id?: string;
  }> {}

const CalendarItemEdit: React.FC<CalendarItemEditProps> = ({
  history,
  match,
}) => {
  const {
    calendarItems,
    saving,
    savingError,
    saveCalendarItem,
    saveCalendarItemOffline,
  } = useContext(CalendarItemContext);
  const [title, setTitle] = useState("");
  const [type, setType] = useState("event");
  const [noOfGuests, setNoOfGuests] = useState(0);
  const [startDate, setStartDate] = useState(new Date());
  const [endDate, setEndDate] = useState(new Date());
  const [isCompleted, setIsCompleted] = useState(false);
  const [doesRepeat, setDoesRepeat] = useState(false);
  const myLocation = useMyLocation();
  const { latitude: lat, longitude: lng } = myLocation.position?.coords || {};
  const { photos, takePhoto, deletePhoto } = usePhotos(match.params.id || "");
  const [photoToMakeAction, setPhotoToMakeAction] = useState<MyPhoto>();
  const [latitude, setLatitude] = useState(lat);
  const [longitude, setLongitude] = useState(lng);
  const [calendarItem, setCalendarItem] = useState<CalendarItemProps>();
  const networkStatus = useContext(NetworkStatusContext);
  useEffect(() => {
    setLatitude(lat);
    setLongitude(lng);
  }, [lat, lng]);
  useEffect(() => {
    log("useEffect");
    const routeId = match.params.id || "";
    const calendarItem = calendarItems?.find((it) => it._id === routeId);
    setCalendarItem(calendarItem);
    if (calendarItem) {
      setTitle(calendarItem.title);
      setType(calendarItem.type);
      setNoOfGuests(calendarItem.noOfGuests);
      setStartDate(new Date(calendarItem.startDate));
      setEndDate(new Date(calendarItem.endDate));
      setIsCompleted(calendarItem.isCompleted);
      setDoesRepeat(calendarItem.doesRepeat);
      setLatitude(calendarItem.latitude);
      setLongitude(calendarItem.longitude);
    }
  }, [match.params.id, calendarItems]);
  const handleSave = useCallback(() => {
    const editedCalendarItem = calendarItem
      ? {
          ...calendarItem,
          title,
          type,
          noOfGuests,
          startDate,
          endDate,
          isCompleted,
          doesRepeat,
          latitude,
          longitude,
        }
      : {
          title,
          type,
          noOfGuests,
          startDate,
          endDate,
          isCompleted,
          doesRepeat,
          latitude,
          longitude,
        };

    if (networkStatus.connected) {
      saveCalendarItem &&
        saveCalendarItem({
          ...editedCalendarItem,
          latitude: latitude as number,
          longitude: longitude as number,
        }).then(() => history.goBack());
    } else {
      saveCalendarItemOffline &&
        saveCalendarItemOffline({
          ...editedCalendarItem,
          latitude: latitude as number,
          longitude: longitude as number,
        }).then(() => history.goBack());
    }
  }, [
    calendarItem,
    saveCalendarItem,
    title,
    type,
    noOfGuests,
    startDate,
    endDate,
    isCompleted,
    doesRepeat,
    latitude,
    longitude,
    history,
  ]);
  log("render");
  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Edit</IonTitle>
          <IonButtons slot="end">
            <IonButton onClick={handleSave}>Save</IonButton>
          </IonButtons>
        </IonToolbar>
      </IonHeader>
      <IonContent>
        <IonLabel>Title: </IonLabel>
        <IonInput
          value={title}
          onIonChange={(e) => setTitle(e.detail.value || "")}
        />
        <IonLabel>Type: </IonLabel>
        <IonSelect
          value={type}
          placeholder="Select Type"
          onIonChange={(e) => setType(e.detail.value)}
        >
          <IonSelectOption value="event" aria-selected="true">
            Event
          </IonSelectOption>
          <IonSelectOption value="task">Task</IonSelectOption>
          <IonSelectOption value="personal">Personal</IonSelectOption>
        </IonSelect>
        <IonLabel>Guests: </IonLabel>
        <IonInput
          type="number"
          value={noOfGuests}
          onIonChange={(e) => setNoOfGuests(parseInt(e.detail.value || "0"))}
        />
        <IonLabel>Start date: </IonLabel>
        <IonDatetime
          value={startDate.toISOString()}
          firstDayOfWeek={1}
          showDefaultButtons={true}
          onIonChange={(e) => {
            setStartDate(new Date(e.detail.value?.toLocaleString() || ""));
          }}
        />
        <IonLabel>End date: </IonLabel>
        <IonDatetime
          value={endDate.toISOString()}
          firstDayOfWeek={1}
          showDefaultButtons={true}
          onIonChange={(e) =>
            setEndDate(new Date(e.detail.value?.toLocaleString() || ""))
          }
        />
        <IonLabel>Is Completed: </IonLabel>
        <IonCheckbox
          checked={isCompleted}
          onIonChange={(e) => setIsCompleted(e.detail.checked)}
        />
        <br />
        <IonLabel>Does Repeat: </IonLabel>
        <IonCheckbox
          checked={doesRepeat}
          onIonChange={(e) => setDoesRepeat(e.detail.checked)}
        />
        {latitude && longitude && (
          <MyMap
            lat={latitude}
            lng={longitude}
            onMapClick={({ latitude, longitude }) => {
              setLatitude(latitude);
              setLongitude(longitude);
            }}
            onMarkerClick={() => log("onMarker")}
          />
        )}
        {match.params.id && (
          <>
            <IonFabButton onClick={() => takePhoto()}>
              <IonIcon icon={camera} />
            </IonFabButton>
            <IonGrid>
              <IonRow>
                {photos.map((photo, index) => (
                  <IonCol size="3" key={index}>
                    <IonImg
                      onClick={() => setPhotoToMakeAction(photo)}
                      src={photo.webviewPath}
                    />
                  </IonCol>
                ))}
              </IonRow>
            </IonGrid>
            <IonActionSheet
              isOpen={!!photoToMakeAction}
              buttons={[
                {
                  text: "Save",
                  role: "save",
                  icon: save,
                  handler: () => {
                    const link = document.createElement("a");
                    link.href = photoToMakeAction?.webviewPath || "";
                    link.download = "downloaded_image.jpg";

                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    setPhotoToMakeAction(undefined);
                  },
                },
                {
                  text: "Delete",
                  role: "destructive",
                  icon: trash,
                  handler: () => {
                    if (photoToMakeAction) {
                      deletePhoto(photoToMakeAction);
                      setPhotoToMakeAction(undefined);
                    }
                  },
                },
                {
                  text: "Cancel",
                  icon: close,
                  role: "cancel",
                },
              ]}
              onDidDismiss={() => setPhotoToMakeAction(undefined)}
            />
          </>
        )}
        <IonLoading isOpen={saving} />
        {savingError && (
          <div>{savingError.message || "Failed to save calendar item"}</div>
        )}
      </IonContent>
    </IonPage>
  );
};

export default CalendarItemEdit;
