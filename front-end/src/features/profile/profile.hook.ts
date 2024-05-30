import { useQuery, useMutation } from '@tanstack/react-query';

import {
  getUserCard,
  getFollowingListData,
  putUserCard,
  postSearchUser,
  postUnfollowUser,
  postFollowUser,
  postUserProfile,
  postEmail,
  postCertificate,
  getRecommendMates,
  getMutualFollowUsers,
  patchProfileSetting,
} from './profile.api';

import { type CardType } from '@/entities/shared-posts-filter';

export const useMutualFollowUsers = (enabled: boolean) =>
  useQuery({
    queryKey: ['/profile/mutual/follow'],
    queryFn: async () =>
      await getMutualFollowUsers().then(res => res.data.data.followingList),
    enabled,
  });

export const useUserProfile = (memberId: string) =>
  useMutation({
    mutationFn: async () => await postUserProfile(memberId),
    onSuccess: data => data.data,
  });

export const useUserCard = (cardId: number) =>
  useQuery({
    queryKey: [`/profile/card/${cardId}`],
    queryFn: async () => await getUserCard(cardId),
    enabled: cardId !== undefined,
  });

export const usePutUserCard = (cardId: number) =>
  useMutation({
    mutationFn: async (data: {
      location: string;
      features: {
        smoking: string;
        roomSharingOption: string;
        mateAge: number;
        options: string;
      };
    }) => await putUserCard(cardId, data.location, data.features),
  });

export const useFollowingListData = () =>
  useQuery({
    queryKey: [`/profile/follow`],
    queryFn: async () => await getFollowingListData().then(res => res),
  });

export const useFollowUser = (memberId: string) =>
  useMutation({
    mutationFn: async () => {
      await postFollowUser(memberId);
    },
  });

export const useUnfollowUser = (memberId: string) =>
  useMutation({
    mutationFn: async () => {
      await postUnfollowUser(memberId);
    },
  });

export const useSearchUser = (email: string) =>
  useMutation({
    mutationFn: async () => await postSearchUser(email),
    onSuccess: data => data.data,
  });

export const useGetCode = (email: string, univName: string) =>
  useMutation({
    mutationFn: async () => await postEmail(email, univName),
  });

export const useCertification = (
  email: string,
  univName: string,
  code: number,
) =>
  useMutation({
    mutationFn: async () => await postCertificate(email, univName, code),
    onSuccess: data => data.data,
  });

export const useRecommendMates = ({
  enabled,
  cardOption,
}: {
  enabled: boolean;
  cardOption: CardType;
}) =>
  useQuery({
    queryKey: ['/profile/recommend', cardOption],
    queryFn: async () => await getRecommendMates(cardOption).then(res => res),
    enabled,
  });

export const useProfileSetting = () =>
  useMutation({
    mutationFn: async (status: boolean) => await patchProfileSetting(status),
    onSuccess: data => data.data,
  });
