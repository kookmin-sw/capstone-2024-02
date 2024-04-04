import { type RentalType, type RoomType } from '@/shared/types';

export interface GetSharedPostsFilter {
  roomType?: RoomType[];
  rentalType?: RentalType[];
}

export interface GetSharedPostsProps {
  filter?: GetSharedPostsFilter;
  search?: string;
  page?: number;
}
