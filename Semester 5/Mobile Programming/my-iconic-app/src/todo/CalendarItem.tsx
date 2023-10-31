import React, { memo } from 'react';
import { IonItem, IonLabel } from '@ionic/react';
import { getLogger } from '../core';
import { CalendarItemProps } from './CalendarItemProps';

const log = getLogger('Item');

interface CalendarItemPropsExt extends CalendarItemProps {
  onEdit: (id?: string) => void;
}

const formatDate = (date: Date) : string => {
  const day = date.getDate().toString().padStart(2, '0');
  const month = (date.getMonth() + 1).toString().padStart(2, '0'); // Month is 0-indexed
  const year = date.getFullYear();
  return `${day}/${month}/${year}`;
};

const CalendarItem: React.FC<CalendarItemPropsExt> = ({ id, title, type, noOfGuests, startDate, endDate, isCompleted, doesRepeat, onEdit }) => {
  return (
    <IonItem onClick={() => onEdit(id)}>
      <IonLabel>Title: {title}</IonLabel>
      <IonLabel>Type: {type}</IonLabel>
      <IonLabel>NoOfGuests: {noOfGuests}</IonLabel>
      <IonLabel>StartDate: {startDate.toString()}</IonLabel>
      <IonLabel>EndDate: {endDate.toString()}</IonLabel>
      <IonLabel>IsCompleted: {isCompleted ? "Yes" : "No"}</IonLabel>
      <IonLabel>DoesRepeat: {doesRepeat ? "Yes" : "No"}</IonLabel>
    </IonItem>
  );
};

export default memo(CalendarItem);
