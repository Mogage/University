import React, { memo } from "react";
import { IonItem, IonLabel } from "@ionic/react";
import { getLogger } from "../core";
import { CalendarItemProps } from "./CalendarItemProps";

interface CalendarItemPropsExt extends CalendarItemProps {
  onEdit: (id?: string) => void;
}

const formatDate = (date: Date): string => {
  const day = date.getDay().toString().padStart(2, "0");
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
  onEdit,
}) => {
  return (
    <IonLabel style={{ "text-align": "center" }} onClick={() => onEdit(_id)}>
      <div>Title: {title}</div>
      <div>Type: {type}</div>
      <div>NoOfGuests: {noOfGuests}</div>
      <div>StartDate: {formatDate(new Date(startDate))}</div>
      <div>EndDate: {formatDate(new Date(endDate))}</div>
      <div>IsCompleted: {isCompleted ? "Yes" : "No"}</div>
      <div>DoesRepeat: {doesRepeat ? "Yes" : "No"}</div>
      <br />
    </IonLabel>
  );
};

export default memo(CalendarItem);
