export type RoomType =
  | 'VILLA_1'
  | 'VILLA_2'
  | 'VILLA_3'
  | 'OFFICE_TEL_1'
  | 'OFFICE_TEL_2'
  | 'OFFICE_TEL_3'
  | 'APT';

export const RoomTypeValue: Record<RoomType, number> = {
  VILLA_1: 0,
  VILLA_2: 1,
  VILLA_3: 2,
  OFFICE_TEL_1: 3,
  OFFICE_TEL_2: 4,
  OFFICE_TEL_3: 5,
  APT: 6,
};
