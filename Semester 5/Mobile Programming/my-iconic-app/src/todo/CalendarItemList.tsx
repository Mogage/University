import React, { useContext, useEffect, useState } from "react";
import { RouteComponentProps } from "react-router";
import {
  IonButton,
  IonButtons,
  IonContent,
  IonFab,
  IonFabButton,
  IonHeader,
  IonIcon,
  IonInfiniteScroll,
  IonInfiniteScrollContent,
  IonList,
  IonLoading,
  IonPage,
  IonSearchbar,
  IonSelect,
  IonSelectOption,
  IonTitle,
  IonToolbar,
} from "@ionic/react";
import { add } from "ionicons/icons";
import CalendarItem from "./CalendarItem";
import { getLogger } from "../core";
import { CalendarItemContext } from "./CalendarItemProvider";
import { usePreferences } from "../hooks/usePreferencesToken";
import { AuthContext } from "../auth";
import { NetworkStatusContext } from "../hooks/NetworkStatusProvider";
import { usePreferencesCalendarItems } from "../hooks/usePreferencesCalendarItems";

const log = getLogger("CalendarItemList");

const CalendarItemList: React.FC<RouteComponentProps> = ({ history }) => {
  const { calendarItems, fetching, fetchingError } =
    useContext(CalendarItemContext);
  const { logout } = useContext(AuthContext);
  const networkStatus = useContext(NetworkStatusContext);
  const [disableInfinteScroll, setDisableInfiniteScroll] =
    useState<boolean>(false);
  const [currentIndex, setCurrentIndex] = useState<number>(7);
  const [searchText, setSearchText] = useState<string>("");
  const [filters, setFilters] = useState<string>("all");
  log("render", fetching);
  usePreferences();
  usePreferencesCalendarItems();

  const handleLogout = () => {
    logout?.();
  };

  async function fetchData() {
    setCurrentIndex(currentIndex + 5);
    if (calendarItems && currentIndex >= calendarItems.length) {
      setDisableInfiniteScroll(true);
      return;
    }
  }

  async function searchNext($event: CustomEvent<void>) {
    await fetchData();
    await ($event.target as HTMLIonInfiniteScrollElement).complete();
  }

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Calendar Item List</IonTitle>
          {networkStatus.connected === false && (
            <div className={"red-circle"}></div>
          )}
          {networkStatus.connected === true && (
            <div className={"green-circle"}></div>
          )}
          <IonButtons slot="end">
            <IonButton onClick={handleLogout}>Logout</IonButton>
          </IonButtons>
        </IonToolbar>
      </IonHeader>
      <IonContent>
        <IonSearchbar
          value={searchText}
          debounce={500}
          onIonInput={(text) => setSearchText(text.detail.value!)}
        />
        <IonSelect
          value={filters}
          placeholder="Select type"
          onIonChange={(e) => setFilters(e.detail.value)}
        >
          <IonSelectOption value="all">All</IonSelectOption>
          <IonSelectOption value="event">Event</IonSelectOption>
          <IonSelectOption value="task">Task</IonSelectOption>
          <IonSelectOption value="personal">Personal</IonSelectOption>
        </IonSelect>
        <IonLoading isOpen={fetching} message="Fetching calendar items" />
        {calendarItems && (
          <IonList lines="none">
            {calendarItems
              .filter(
                (calendarItem) =>
                  calendarItem.title
                    .toLowerCase()
                    .includes(searchText.toLocaleLowerCase()) &&
                  (filters === "all" ? true : calendarItem.type === filters)
              )
              .sort((a, b) => {
                a._id?.localeCompare(b._id!);
              })
              .slice(0, currentIndex)
              .map(
                ({
                  _id,
                  title,
                  type,
                  noOfGuests,
                  startDate,
                  endDate,
                  isCompleted,
                  doesRepeat,
                }) => (
                  <CalendarItem
                    key={_id}
                    _id={_id}
                    title={title}
                    type={type}
                    noOfGuests={noOfGuests}
                    startDate={startDate}
                    endDate={endDate}
                    isCompleted={isCompleted}
                    doesRepeat={doesRepeat}
                    onEdit={(id) => history.push(`/calendarItem/${id}`)}
                  />
                )
              )}
          </IonList>
        )}
        <IonInfiniteScroll
          threshold="-70px"
          disabled={disableInfinteScroll}
          onIonInfinite={(e: CustomEvent<void>) => searchNext(e)}
        >
          <IonInfiniteScrollContent></IonInfiniteScrollContent>
        </IonInfiniteScroll>
        {fetchingError && (
          <div>{fetchingError.message || "Failed to fetch calendar items"}</div>
        )}
        <IonFab vertical="bottom" horizontal="end" slot="fixed">
          <IonFabButton onClick={() => history.push("/calendarItem")}>
            <IonIcon icon={add} />
          </IonFabButton>
        </IonFab>
      </IonContent>
    </IonPage>
  );
};

export default CalendarItemList;
