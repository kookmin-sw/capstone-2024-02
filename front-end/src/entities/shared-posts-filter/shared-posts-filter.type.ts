export type CardType = 'my' | 'mate';
export type DealType = '전세' | '월세';
export type RoomType = '원룸' | '빌라/투룸이상' | '아파트' | '오피스텔';
export type RoomCountType = '1개' | '2개' | '3개 이상';
export type FloorType = '지상' | '반지하' | '옥탑';
export type ExtraInfoType =
  | '주차 가능'
  | '에어컨'
  | '냉장고'
  | '세탁기'
  | '엘리베이터'
  | '베란다/테라스'
  | '복층형';

export interface SharedPostsFilter {
  cardType?: CardType;
  roomInfo: {
    roomType?: RoomType;
    hasLivingRoom: boolean;
    roomCount?: RoomCountType;
    restRoomCount?: RoomCountType;
    size?: { low: number; high: number };
    floor?: FloorType;
  };
  dealInfo?: {
    dealType?: DealType;
    expectedFee?: { low: number; high: number };
  };
  extraInfo: Partial<Record<ExtraInfoType, boolean>>;
}

export type SharedPostsFilterType = keyof SharedPostsFilter;

export const SharedPostsFilterTypeValue: Record<SharedPostsFilterType, string> =
  {
    cardType: '메이트 카드',
    dealInfo: '거래 정보',
    roomInfo: '방 정보',
    extraInfo: '추가 정보',
  };

export type SharedPostsType = 'hasRoom' | 'homeless' | 'dormitory';
