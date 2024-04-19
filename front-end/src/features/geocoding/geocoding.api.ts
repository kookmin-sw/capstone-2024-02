import axios from 'axios';

import { type FromAddrToCoordDTO } from '@/features/geocoding/geocoding.type';

export const fromAddrToCoord = async ({ query }: { query: string }) =>
  await axios.get<FromAddrToCoordDTO>(`/api/geocode/${query}`);

export const getGeolocation = ({
  onSuccess,
  onError,
}: {
  onSuccess: (position: GeolocationPosition) => void;
  onError: (error: GeolocationPositionError) => void;
}) => {
  if (navigator.geolocation != null) {
    navigator.geolocation.getCurrentPosition(onSuccess, onError);
  }
};
