import React, { useContext } from 'react';
import { RouteComponentProps } from 'react-router';
import {
  IonContent,
  IonFab,
  IonFabButton,
  IonHeader,
  IonIcon,
  IonList, IonLoading,
  IonPage,
  IonTitle,
  IonToolbar
} from '@ionic/react';
import { add } from 'ionicons/icons';
import CalendarItem from './CalendarItem';
import { getLogger } from '../core';
import { CalendarItemContext } from './CalendarItemProvider';

const log = getLogger('ItemList');

const CalendarItemList: React.FC<RouteComponentProps> = ({ history }) => {
  const { calendarItems, fetching, fetchingError } = useContext(CalendarItemContext);
  log('render');
  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>My App</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent>
        <IonLoading isOpen={fetching} message="Fetching items" />
        {calendarItems && (
          <IonList>
            {calendarItems.map(({ id, title, type, noOfGuests, startDate, endDate, isCompleted, doesRepeat}) =>
              <CalendarItem key={id} id={id} title={title} type={type} 
                            noOfGuests={noOfGuests}
                            startDate={startDate} endDate={endDate} 
                            isCompleted={isCompleted} doesRepeat={doesRepeat}
                            onEdit={id => history.push(`/calendarItem/${id}`)} />)}
          </IonList>
        )}
        {fetchingError && (
          <div>{fetchingError.message || 'Failed to fetch calendar items'}</div>
        )}
        <IonFab vertical="bottom" horizontal="end" slot="fixed">
          <IonFabButton onClick={() => history.push('/calendarItem')}>
            <IonIcon icon={add} />
          </IonFabButton>
        </IonFab>
      </IonContent>
    </IonPage>
  );
};

export default CalendarItemList;
