import axios from 'axios';

import {
  type GetUserProfileDTO,
  type GetUserCardDTO,
  type PutUserCardDTO,
  type PostFollowDTO,
  type GetFollowingListDTO,
} from './profile.dto';

export const getUserProfileData = async (memberId: string) =>
  await axios
    .get<GetUserProfileDTO>(`/maru-api/profile/${memberId}`)
    .then(res => res.data);

export const getUserCard = async (cardId: number) =>
  await axios
    .get<GetUserCardDTO>(`/maru-api/profile/card/${cardId}`)
    .then(res => res.data);

export const putUserCard = async (
  cardId: number,
  location: string,
  features: Array<string | undefined>,
) =>
  await axios
    .put<PutUserCardDTO>(`/maru-api/profile/${cardId}`, {
      location,
      features,
    })
    .then(res => {
      console.log(res.data);
      return res.data;
    });

export const getFollowingListData = async () =>
  await axios
    .get<GetFollowingListDTO>(`/maru-api/profile/follow`)
    .then(res => res.data);

export const postFollowData = async (memberId: string) => {
  await axios
    .post<PostFollowDTO>(`/maru-api/profile/${memberId}/follow`, {})
    .then(res => res.data);
};
