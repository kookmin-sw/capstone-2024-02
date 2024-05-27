import { useMemo } from 'react';
import { useRecoilState, useResetRecoilState } from 'recoil';

import { sharedPostsFilterState } from './shared-posts-filter.atom';
import {
  type RoomType,
  RoomTypeValue,
  type SharedPostsFilter,
  type DealType,
  DealTypeValue,
  type FloorType,
  FloorTypeValue,
} from './shared-posts-filter.type';

export const useSharedPostsFilter = () => {
  const [filter, setFilter] = useRecoilState<SharedPostsFilter>(
    sharedPostsFilterState,
  );

  const reset = useResetRecoilState(sharedPostsFilterState);

  const derivedFilter = useMemo<{
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
  }>(() => {
    let numberOfRoom: number | undefined;
    if (filter.roomInfo.roomCount === '1개') numberOfRoom = 1;
    else if (filter.roomInfo.roomCount === '2개') numberOfRoom = 2;
    else if (filter.roomInfo.roomCount === '3개 이상') numberOfRoom = 3;

    let numberOfRestRoom: number | undefined;
    if (filter.roomInfo.restRoomCount === '1개') numberOfRestRoom = 1;
    else if (filter.roomInfo.restRoomCount === '2개') numberOfRestRoom = 2;
    else if (filter.roomInfo.restRoomCount === '3개 이상') numberOfRestRoom = 3;

    const isRoomType = (option: string): option is RoomType =>
      Object.keys(RoomTypeValue).includes(option);
    const roomTypes: number[] =
      filter.roomInfo.roomType != null
        ? Object.entries(filter.roomInfo.roomType)
            .filter(([option, selected]) => selected)
            .reduce<number[]>((prev, [option, selected]) => {
              if (isRoomType(option)) prev.push(RoomTypeValue[option]);
              return prev;
            }, [])
        : [];

    const isDealType = (option: string): option is DealType =>
      Object.keys(DealTypeValue).includes(option);
    const rentalTypes: number[] =
      filter.dealInfo.dealType != null
        ? Object.entries(filter.dealInfo.dealType)
            .filter(([option, selected]) => selected)
            .reduce<number[]>((prev, [option, selected]) => {
              if (isDealType(option)) prev.push(DealTypeValue[option]);
              return prev;
            }, [])
        : [];

    const isFloorType = (option: string): option is FloorType =>
      Object.keys(FloorTypeValue).includes(option);
    const floorTypes: number[] =
      filter.roomInfo.floor != null
        ? Object.entries(filter.roomInfo.floor)
            .filter(([option, selected]) => selected)
            .reduce<number[]>((prev, [option, selected]) => {
              if (isFloorType(option)) prev.push(FloorTypeValue[option]);
              return prev;
            }, [])
        : [];

    return {
      roomTypes: roomTypes.length === 0 ? undefined : roomTypes,
      rentalTypes: rentalTypes.length === 0 ? undefined : rentalTypes,
      expectedPaymentRange:
        filter.dealInfo?.expectedFee != null
          ? {
              start: filter.dealInfo.expectedFee.low,
              end: filter.dealInfo.expectedFee.high,
            }
          : undefined,
      hasLivingRoom: filter.roomInfo.hasLivingRoom,
      numberOfRoom,
      numberOfRestRoom,
      roomSizeRange:
        filter.roomInfo.size != null
          ? {
              start: filter.roomInfo.size.low,
              end: filter.roomInfo.size.high,
            }
          : undefined,
      floorTypes: floorTypes.length === 0 ? undefined : floorTypes,
      canPark: filter.extraInfo.주차가능,
      hasAirConditioner: filter.extraInfo.에어컨,
      hasRefrigerator: filter.extraInfo.냉장고,
      hasWasher: filter.extraInfo.세탁기,
      hasTerrace: filter.extraInfo['베란다/테라스'],
    };
  }, [filter]);

  return useMemo(
    () => ({
      filter,
      derivedFilter,
      setFilter,
      reset,
    }),
    [filter, derivedFilter, setFilter, reset],
  );
};
