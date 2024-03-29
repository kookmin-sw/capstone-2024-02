import { type Address } from '../address';

import { type RoomType, type RentalType } from '@/shared/types';

export interface RoomInfo {
  id: number;
  address: Address;
  roomType: RoomType;
  size: number;
  numberOfRoom: number;
  rentalType: RentalType;
  price: number;
  managementFee: number;
  expectedPayment: number;
}
