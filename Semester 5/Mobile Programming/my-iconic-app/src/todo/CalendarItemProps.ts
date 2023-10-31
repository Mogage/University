export interface CalendarItemProps {
  id?: string;
  title: string;
  type: string;
  noOfGuests: number;
  startDate: Date;
  endDate: Date;
  isCompleted: boolean;
  doesRepeat: boolean;
}
