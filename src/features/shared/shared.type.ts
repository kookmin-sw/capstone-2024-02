import { type RentalType, type RoomType } from '@/shared/types';

export interface GetSharedPostsFilter {
  roomType?: RoomType[];
  rentalType?: RentalType[];
}

export interface GetSharedPostsProps {
  filter?: GetSharedPostsFilter;
  search?: string;
}

export interface PostContentObject {
  id: number;
  title: string;
  content: string;
  thumbnail: {
    id: number;
    storeImagePath: string;
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
  createdAt: Date;
  createdBy: string;
  modifiedAt: Date;
  modifiedBy: string;
}

export interface SortObject {
  empty: boolean;
  unsorted: boolean;
  sorted: boolean;
}

export interface PageableObject {
  pageNumber: number;
  pageSize: number;
  sort: SortObject;
  offset: number;
  paged: boolean;
  unpaged: boolean;
}
