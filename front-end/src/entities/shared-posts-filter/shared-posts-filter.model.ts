export interface SharedPostsFilter {
  userCard?: string;
  mateCard?: string;
  roomInfo?: string;
  dealInfo?: string;
  extraInfo?: string;
}

export type SharedPostsFilterType = keyof SharedPostsFilter;

export const SharedPostsFilterTypeValue: Record<SharedPostsFilterType, string> =
  {
    userCard: '마이카드',
    mateCard: '메이트카드',
    dealInfo: '거래 정보',
    roomInfo: '방 정보',
    extraInfo: '추가 정보',
  };

export type SharedPostsType = 'hasRoom' | 'homeless' | 'dormitory';
