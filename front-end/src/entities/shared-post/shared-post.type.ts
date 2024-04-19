import { type RentalType, type RoomType } from '@/shared/types';

export interface SharedPostListItem {
  id: number;
  title: string;
  content: string;
  thumbnail: {
    fileName: string;
    isThumbnail: boolean;
    order: number;
  };
  publisherAccount: {
    memberId: string;
    email: string;
    nickname: string;
    birthYear: string;
    gender: string;
    phoneNumber: string;
    createdAt: Date;
    createdBy: string;
    modifiedAt: Date;
    modifiedBy: string;
  };
  roomInfo: {
    id: number;
    address: {
      city: string;
      oldAddress: string;
      roadAddress: string;
      detailAddress?: string;
      stationName: string;
      stationTime: number;
      busStopTime: number;
      schoolName: string;
      schoolTime: number;
      convenienceStortTime: number;
    };
    roomType: string;
    size: number;
    numberOfRoom: number;
    rentalType: string;
    price: number;
    managementFee: number;
    expectedPayment: number;
    monthlyFee: number;
  };
  isScrapped: boolean;
  createdAt: Date;
  createdBy: string;
  modifiedAt: Date;
  modifiedBy: string;
}

export interface SharedPost {
  id: number;
  title: string;
  content: string;
  roomImages: Set<{
    fileName: string;
    isThumbnail: boolean;
    order: number;
  }>;
  publisherAccount: {
    memberId: string;
    email: string;
    nickname: string;
    birthYear: string;
    gender: string;
    phoneNumber: string;
    myCardFeatures: string[];
    mateCardFeatures: string[];
    createdAt: Date;
    createdBy: string;
    modifiedAt: Date;
    modifiedBy: string;
  };
  roomInfo: {
    id: number;
    address: {
      city: string;
      oldAddress: string;
      roadAddress: string;
      detailAddress?: string;
      stationName: string;
      stationTime: number;
      busStopTime: number;
      schoolName: string;
      schoolTime: number;
      convenienceStortTime: number;
    };
    roomType: RoomType;
    size: number;
    numberOfRoom: number;
    rentalType: RentalType;
    price: number;
    managementFee: number;
    expectedPayment: number;
    monthlyFee: number;
  };
  isScrapped: boolean;
  scrapCount: number;
  createdAt: Date;
  createdBy: string;
  modifiedAt: Date;
  modifiedBy: string;
}
