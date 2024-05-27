import { type SelectedExtraOptions } from '@/features/shared';

export type AdditionalInfoType =
  | '주차가능'
  | '에어컨'
  | '냉장고'
  | '세탁기'
  | '베란다/테라스';
export const AdditionalInfoTypeValue: Record<
  AdditionalInfoType,
  keyof SelectedExtraOptions
> = {
  주차가능: 'canPark',
  에어컨: 'hasAirConditioner',
  냉장고: 'hasRefrigerator',
  세탁기: 'hasWasher',
  '베란다/테라스': 'hasTerrace',
} as const;

export type CardType = 'my' | 'mate';

export type CountType = '1개' | '2개' | '3개 이상';
export const CountTypeValue: Record<CountType, number> = {
  '1개': 1,
  '2개': 2,
  '3개 이상': 3,
} as const;

export type DealType = '전세' | '월세';
export const DealTypeValue: Record<DealType, number> = {
  월세: 0,
  전세: 1,
} as const;

export type FloorType = '지상' | '반지하' | '옥탑';
export const FloorTypeValue: Record<FloorType, number> = {
  지상: 0,
  반지하: 1,
  옥탑: 2,
} as const;

export type LivingRoomType = '유' | '무';
export const LivingRoomTypeValue: Record<LivingRoomType, boolean> = {
  유: true,
  무: false,
} as const;

export type RoomType = '원룸' | '빌라/투룸이상' | '아파트' | '오피스텔';
export const RoomTypeValue: Record<RoomType, number> = {
  원룸: 0,
  '빌라/투룸이상': 1,
  아파트: 2,
  오피스텔: 3,
} as const;

export interface SharedPostsFilter {
  cardType?: CardType;
  roomInfo: {
    roomType?: Partial<Record<RoomType, boolean>>;
    hasLivingRoom?: boolean;
    roomCount?: CountType;
    restRoomCount?: CountType;
    size?: { low: number; high: number };
    floor?: Partial<Record<FloorType, boolean>>;
  };
  dealInfo: {
    dealType?: Partial<Record<DealType, boolean>>;
    expectedFee?: { low: number; high: number };
  };
  extraInfo: Partial<Record<AdditionalInfoType, boolean>>;
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
