export interface GetSharedPostsFilter {
  roomTypes?: number[];
  rentalTypes?: number[];
  expectedPaymentRange?: { start: number; end: number };
  hasLivingRoom?: boolean;
  numberOfRoom?: number;
  numberOfRestRoom?: number;
  roomSizeRange?: { start: number; end: number };
  floorTypes?: number[];
  canPark?: boolean;
  hasAirConditioner?: boolean;
  hasRefrigerator?: boolean;
  hasWasher?: boolean;
  hasTerrace?: boolean;
}

export interface GetSharedPostsProps {
  filter?: GetSharedPostsFilter;
  search?: string;
  cardOption: 'my' | 'mate';
  page: number;
}

export interface SelectedOptions {
  budget?: string;
  roomType?: string;
  livingRoom?: string;
  roomCount?: string;
  restRoomCount?: string;
  floorType?: string;
}

export interface SelectedExtraOptions {
  canPark?: boolean;
  hasAirConditioner?: boolean;
  hasRefrigerator?: boolean;
  hasTerrace?: boolean;
  hasWasher?: boolean;
}

export interface ImageFile {
  url: string;
  uploaded: boolean;
  file?: File;
  extension?: string;
}

export interface SharedPostProps {
  imageFilesData: Array<{
    fileName: string;
    isThumbNail: boolean;
    order: number;
  }>;
  postData: {
    title: string;
    content: string;
  };
  transactionData?: {
    rentalType: number;
    expectedPayment: number;
  };
  roomDetailData?: {
    roomType: number;
    floorType: number;
    size: number;
    numberOfRoom: number;
    numberOfBathRoom: number;
    hasLivingRoom: boolean;
    recruitmentCapacity: number;
    extraOption: {
      canPark: boolean;
      hasAirConditioner: boolean;
      hasRefrigerator: boolean;
      hasWasher: boolean;
      hasTerrace: boolean;
    };
  };
  locationData: {
    oldAddress: string;
    roadAddress: string;
  };
  roomMateCardData: {
    location: string;
    features: {
      smoking: string;
      roomSharingOption: string;
      mateAge: number | undefined; // 0 ~ 10: +- 범위 값, null: 상관 없어요.
      options: string;
    };
  };
  participationData: {
    recruitmentCapacity: number;
    participationMemberIds: string[];
  };
}
