import React, { useCallback, useContext, useEffect, useState } from 'react';
import {
  IonButton,
  IonButtons,
  IonCheckbox,
  IonContent,
  IonDatetime,
  IonHeader,
  IonInput,
  IonLabel,
  IonLoading,
  IonPage,
  IonTitle,
  IonToolbar
} from '@ionic/react';
import { getLogger } from '../core';
import { CalendarItemContext } from './CalendarItemProvider';
import { RouteComponentProps } from 'react-router';
import { CalendarItemProps } from './CalendarItemProps';

const log = getLogger('ItemEdit');

interface CalendarItemEditProps extends RouteComponentProps<{
  id?: string;
}> {}

const CalendarItemEdit: React.FC<CalendarItemEditProps> = ({ history, match }) => {
  const { calendarItems, saving, savingError, saveCalendarItem } = useContext(CalendarItemContext);
  const [title, setTitle] = useState('');
  const [type, setType] = useState('');
  const [noOfGuests, setNoOfGuests] = useState(0);
  const [startDate, setStartDate] = useState(new Date());
  const [endDate, setEndDate] = useState(new Date());
  const [isCompleted, setIsCompleted] = useState(false);
  const [doesRepeat, setDoesRepeat] = useState(false);
  const [calendarItem, setCalendarItem] = useState<CalendarItemProps>();
  useEffect(() => {
    log('useEffect');
    const routeId = match.params.id || '';
    const calendarItem = calendarItems?.find(it => it.id === routeId);
    setCalendarItem(calendarItem);
    if (calendarItem) {
      setTitle(calendarItem.title);
      setType(calendarItem.type);
      setNoOfGuests(calendarItem.noOfGuests);
      setStartDate(calendarItem.startDate);
      setEndDate(calendarItem.endDate);
      setIsCompleted(calendarItem.isCompleted);
      setDoesRepeat(calendarItem.doesRepeat);
    }
  }, [match.params.id, calendarItems]);
  const handleSave = useCallback(() => {
    const editedCalendarItem = calendarItem ? 
                        { ...calendarItem, title, type, noOfGuests, startDate, endDate, isCompleted, doesRepeat } : 
                        { title, type, noOfGuests, startDate, endDate, isCompleted, doesRepeat };
    saveCalendarItem && saveCalendarItem(editedCalendarItem).then(() => history.goBack());
  }, [calendarItem, saveCalendarItem, title, type, noOfGuests, startDate, endDate, isCompleted, doesRepeat, history]);
  log('render');
  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Edit</IonTitle>
          <IonButtons slot="end">
            <IonButton onClick={handleSave}>
              Save
            </IonButton>
          </IonButtons>
        </IonToolbar>
      </IonHeader>
      <IonContent>
        <IonLabel>Title: </IonLabel>
        <IonInput value={title} onIonChange={e => setTitle(e.detail.value || '')} />
        <IonLabel>Type: </IonLabel>
        <IonInput value={type} onIonChange={e => setType(e.detail.value || '')} />
        <IonLabel>Guests: </IonLabel>
        <IonInput type="number" value={noOfGuests} onIonChange={e => setNoOfGuests(parseInt(e.detail.value || '0'))} />
        <IonLabel>Start date: </IonLabel>
        <IonInput value={startDate.toString()} onIonChange={e => setStartDate(new Date(e.detail.value || ''))} />
        {/* <IonDatetime  value={startDate.toString()} onIonChange={e => setStartDate(new Date(e.detail.value || ''))} /> */}
        <IonLabel>End date: </IonLabel>
        <IonInput value={endDate.toString()} onIonChange={e => setEndDate(new Date(e.detail.value || ''))} />
        {/* <IonDatetime value={endDate.toString()} onIonChange={e => setEndDate(new Date(e.detail.value || ''))} /> */}
        <IonLabel>Is Completed: </IonLabel>
        <IonCheckbox checked={isCompleted} onIonChange={e => setIsCompleted(e.detail.checked)} />
        <br />
        <IonLabel>Does Repeat: </IonLabel>
        <IonCheckbox checked={doesRepeat} onIonChange={e => setDoesRepeat(e.detail.checked)} />
        <IonLoading isOpen={saving} />
        {savingError && (
          <div>{savingError.message || 'Failed to save calendar item'}</div>
        )}
      </IonContent>
    </IonPage>
  );
};

export default CalendarItemEdit;
