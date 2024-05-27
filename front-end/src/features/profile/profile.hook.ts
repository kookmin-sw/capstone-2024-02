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
} from './profile.api';

import { type CardType } from '@/entities/shared-posts-filter';

export const useUserProfile = (memberId: string) =>
  useMutation({
    mutationFn: async () => await postUserProfile(memberId),
    onSuccess: data => data.data,
    onError: error => error,
  });

export const useUserCard = (cardId: number) =>
  useQuery({
    queryKey: [`/api/profile/card/${cardId}`],
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
    queryKey: [`/api/profile/follow`],
    queryFn: getFollowingListData,
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
    queryKey: ['/api/profile/recommend', cardOption],
    queryFn: async () => await getRecommendMates(cardOption),
    enabled,
  });
