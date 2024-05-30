import { useMutation, useQuery } from '@tanstack/react-query';
import { type AxiosResponse } from 'axios';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRecoilState, useResetRecoilState } from 'recoil';

import {
  createDormitorySharedPost,
  createSharedPost,
  deleteDormitorySharedPost,
  deleteSharedPost,
  getDormitorySharedPost,
  getDormitorySharedPosts,
  getSharedPost,
  getSharedPosts,
  scrapDormitoryPost,
  scrapPost,
  updateSharedPost,
} from './shared.api';
import { sharedPostPropState } from './shared.atom';
import {
  type GetDormitorySharedPostDTO,
  type GetSharedPostDTO,
} from './shared.dto';
import {
  type GetSharedPostsProps,
  type SelectedExtraOptions,
  type SelectedOptions,
  type SharedPostProps,
} from './shared.type';
import { fromAddrToCoord } from '../geocoding';

import { useAuthValue, useUserData } from '@/features/auth';
import { useDebounce } from '@/shared/debounce';
import { type FailureDTO, type SuccessBaseDTO } from '@/shared/types';

export const usePaging = ({
  totalPages,
  sliceSize,
}: {
  totalPages: number;
  sliceSize: number;
}) => {
  const [page, setPage] = useState(1);

  const currentSlice = useMemo(
    () => Math.floor((page - 1) / sliceSize),
    [page, sliceSize],
  );
  const sliceCount = useMemo(
    () => Math.floor(totalPages / sliceSize),
    [totalPages, sliceSize],
  );

  const isFirstPage = useMemo(() => page === 1, [page]);
  const isLastPage = useMemo(() => page === totalPages, [page, totalPages]);

  const handleSetPage = useCallback(
    (newPage: number) => {
      if (page < 0 || page > totalPages) return;
      setPage(newPage);
    },
    [page, totalPages],
  );

  const handleNextPage = useCallback(() => {
    if (isLastPage) return;
    setPage(prev => prev + 1);
  }, [isLastPage, setPage]);

  const handlePrevPage = useCallback(() => {
    if (isFirstPage) return;
    setPage(prev => prev - 1);
  }, [isFirstPage, setPage]);

  return useMemo(
    () => ({
      page,
      totalPages,
      sliceSize,
      currentSlice,
      sliceCount,
      isFirstPage,
      isLastPage,
      handleSetPage,
      handleNextPage,
      handlePrevPage,
    }),
    [
      page,
      totalPages,
      sliceSize,
      currentSlice,
      sliceCount,
      isFirstPage,
      isLastPage,
      handleSetPage,
      handleNextPage,
      handlePrevPage,
    ],
  );
};

export const useSharedPostProps = ({
  postId,
  type,
}: {
  postId?: number;
  type?: 'hasRoom' | 'dormitory';
}) => {
  const [state, setState] = useRecoilState(sharedPostPropState);
  const reset = useResetRecoilState(sharedPostPropState);

  useEffect(() => {
    reset();

    if (postId == null) return;

    const setStateWithPost = ({
      data,
    }: GetSharedPostDTO | GetDormitorySharedPostDTO) => {
      fromAddrToCoord({ query: data.address.roadAddress })
        .then(res => {
          const address = res.shift();
          if (address != null) setState(prev => ({ ...prev, address }));
        })
        .catch(err => {
          console.error(err);
        });

      let roomCount = '1개';
      let restRoomCount = '1개';
      if ('roomInfo' in data) {
        if (data.roomInfo.numberOfRoom === 2) roomCount = '2개';
        else if (data.roomInfo.numberOfRoom === 3) roomCount = '3개 이상';

        if (data.roomInfo.numberOfBathRoom === 2) restRoomCount = '2개';
        else if (data.roomInfo.numberOfBathRoom === 3) restRoomCount = '3개';
      }

      if ('roomInfo' in data) {
        setState({
          ...state,
          title: data.title,
          content: data.content,
          images: data.roomImages.map(({ fileName }) => ({
            url: fileName,
            uploaded: true,
          })),
          mates: data.participants.reduce<
            Record<
              string,
              { memberId: string; nickname: string; profileImage: string }
            >
          >(
            (
              prev,
              { memberId, nickname, profileImageFileName: profileImage },
            ) => {
              const next = { ...prev };
              next[memberId] = { memberId, nickname, profileImage };
              return next;
            },
            {},
          ),
          mateLimit: data.roomInfo.recruitmentCapacity,
          expectedMonthlyFee: data.roomInfo.expectedPayment,
          houseSize: data.roomInfo.size,
          selectedOptions: {
            roomType: data.roomInfo.roomType,
            roomCount,
            budget: data.roomInfo.rentalType,
            floorType: data.roomInfo.floorType,
            livingRoom: data.roomInfo.hasLivingRoom ? '유' : '무',
            restRoomCount,
          },
          selectedExtraOptions: {
            canPark: data.roomInfo.extraOption.canPark,
            hasAirConditioner: data.roomInfo.extraOption.hasAirConditioner,
            hasRefrigerator: data.roomInfo.extraOption.hasRefrigerator,
            hasWasher: data.roomInfo.extraOption.hasWasher,
            hasTerrace: data.roomInfo.extraOption.hasTerrace,
          },
          mateCard: {
            ...state.mateCard,
            location: data.address.roadAddress,
            features: {
              smoking: data.roomMateFeatures.smoking,
              roomSharingOption: data.roomMateFeatures.roomSharingOption,
              mateAge:
                data.roomMateFeatures.mateAge != null
                  ? +data.roomMateFeatures.mateAge
                  : undefined,
              options: new Set(
                JSON.parse(data.roomMateFeatures.options) as string[],
              ),
            },
          },
        });
      } else {
        setState({
          ...state,
          title: data.title,
          content: data.content,
          images: data.roomImages.map(({ fileName }) => ({
            url: fileName,
            uploaded: true,
          })),
          mates: data.participants.reduce<
            Record<
              string,
              { memberId: string; nickname: string; profileImage: string }
            >
          >(
            (
              prev,
              { memberId, nickname, profileImageFileName: profileImage },
            ) => {
              const next = { ...prev };
              next[memberId] = { memberId, nickname, profileImage };
              return next;
            },
            {},
          ),
          mateLimit: data.recruitmentCapacity,
          expectedMonthlyFee: 0,
          houseSize: 0,
          selectedOptions: {},
          selectedExtraOptions: {},
          mateCard: {
            ...state.mateCard,
            location: data.address.roadAddress,
            features: {
              smoking: data.roomMateFeatures.smoking,
              roomSharingOption: data.roomMateFeatures.roomSharingOption,
              mateAge:
                data.roomMateFeatures.mateAge != null
                  ? +data.roomMateFeatures.mateAge
                  : undefined,
              options: new Set(
                JSON.parse(data.roomMateFeatures.options) as string[],
              ),
            },
          },
        });
      }
    };

    (async () => {
      const post =
        type === 'hasRoom'
          ? await getSharedPost(postId)
          : await getDormitorySharedPost(postId);

      setStateWithPost(post.data);
    })();
  }, []);

  const handleOptionClick = (
    optionName: keyof SelectedOptions,
    item: string,
  ) => {
    setState(prev => ({
      ...prev,
      selectedOptions: {
        ...prev.selectedOptions,
        [optionName]: prev.selectedOptions[optionName] === item ? null : item,
      },
    }));
  };

  const handleExtraOptionClick = (option: keyof SelectedExtraOptions) => {
    setState(prev => {
      const value = prev.selectedExtraOptions[option] ?? false;
      return {
        ...prev,
        selectedExtraOptions: {
          ...prev.selectedExtraOptions,
          [option]: !value,
        },
      };
    });
  };

  const isOptionSelected = (optionName: keyof SelectedOptions, item: string) =>
    state.selectedOptions[optionName] === item;

  const isExtraOptionSelected = (item: keyof SelectedExtraOptions) =>
    state.selectedExtraOptions[item] === true;

  const handleMateCardEssentialFeatureChange = (
    key: 'smoking' | 'roomSharingOption' | 'mateAge',
    value: string | number | undefined,
  ) => {
    setState(prev => {
      if (prev.mateCard.features[key] === value) {
        const newFeatures = { ...prev.mateCard.features };
        newFeatures[key] = undefined;
        return {
          ...prev,
          mateCard: { ...prev.mateCard, features: newFeatures },
        };
      }
      return {
        ...prev,
        mateCard: {
          ...prev.mateCard,
          features: { ...prev.mateCard.features, [key]: value },
        },
      };
    });
  };

  const handleMateCardOptionalFeatureChange = (option: string) => {
    setState(prev => {
      const { options } = prev.mateCard.features;
      const newOptions = new Set(options);

      if (options.has(option)) newOptions.delete(option);
      else newOptions.add(option);

      return {
        ...prev,
        mateCard: {
          ...prev.mateCard,
          features: { ...prev.mateCard.features, options: newOptions },
        },
      };
    });
  };

  const derivedMateCardFeatures = useMemo(() => {
    const options: string[] = [];
    const { features } = state.mateCard;
    features.options.forEach(option => options.push(option));

    return {
      smoking: features?.smoking ?? '상관없어요',
      roomSharingOption: features?.roomSharingOption ?? '상관없어요',
      mateAge: state.mateCard.birthYear,
      options: JSON.stringify(options),
    };
  }, [state.mateCard]);

  const auth = useAuthValue();
  const { data: user } = useUserData(auth?.accessToken != null);

  useEffect(() => {
    if (user?.gender != null) {
      setState(prev => ({
        ...prev,
        mateCard: {
          ...prev.mateCard,
          gender: user.gender,
        },
      }));
    }
  }, [user, setState]);

  const isCreatable = useMemo(
    () =>
      state.title.trim().length > 0 &&
      state.content.trim().length > 0 &&
      state.images.length >= 2 &&
      state.address != null &&
      state.mateLimit > 0 &&
      state.expectedMonthlyFee > 0 &&
      state.houseSize > 0 &&
      state.selectedOptions.budget != null &&
      state.selectedOptions.roomType != null &&
      state.selectedOptions.livingRoom != null &&
      state.selectedOptions.roomCount != null &&
      state.selectedOptions.restRoomCount != null &&
      state.selectedOptions.floorType != null,
    [state],
  );

  return {
    ...state,
    derivedMateCardFeatures,
    isCreatable,
    setSharedPostProps: setState,
    reset,
    handleOptionClick,
    handleExtraOptionClick,
    isOptionSelected,
    isExtraOptionSelected,
    handleMateCardOptionalFeatureChange,
    handleMateCardEssentialFeatureChange,
  };
};

export const useCreateSharedPost = () =>
  useMutation<AxiosResponse<SuccessBaseDTO>, FailureDTO, SharedPostProps>({
    mutationFn: createSharedPost,
  });

export const useSharedPosts = ({
  filter,
  search,
  cardOption,
  page,
  enabled,
}: GetSharedPostsProps & { enabled: boolean }) => {
  const debounceFilter = useDebounce(filter, 1000);

  return useQuery({
    queryKey: [
      '/shared/posts/studio',
      { cardOption, debounceFilter, search, page },
    ],
    queryFn: async () =>
      await getSharedPosts({
        cardOption,
        filter: debounceFilter,
        search,
        page,
      }).then(res => res),
    enabled,
  });
};

export const useSharedPost = ({
  postId,
  enabled,
}: {
  postId: number;
  enabled: boolean;
}) =>
  useQuery({
    queryKey: [`/shared/posts/studio/${postId}`],
    queryFn: async () =>
      await getSharedPost(postId).then(response => response.data),
    enabled,
  });

export const useUpdateSharedPost = () =>
  useMutation<
    AxiosResponse<SuccessBaseDTO>,
    FailureDTO,
    { postId: number; postData: SharedPostProps }
  >({
    mutationFn: updateSharedPost,
  });

export const useDeleteSharedPost = () =>
  useMutation<AxiosResponse<SuccessBaseDTO>, FailureDTO, number>({
    mutationFn: deleteSharedPost,
  });

export const useScrapSharedPost = () =>
  useMutation<AxiosResponse<SuccessBaseDTO>, FailureDTO, number>({
    mutationFn: scrapPost,
  });

export const useCreateDormitorySharedPost = () =>
  useMutation<AxiosResponse<SuccessBaseDTO>, FailureDTO, SharedPostProps>({
    mutationFn: createDormitorySharedPost,
  });

export const useDormitorySharedPosts = ({
  filter,
  search,
  cardOption,
  page,
  enabled,
}: GetSharedPostsProps & { enabled: boolean }) => {
  const debounceFilter = useDebounce(filter, 1000);

  return useQuery({
    queryKey: [
      '/shared/posts/dormitory',
      { cardOption, debounceFilter, search, page },
    ],
    queryFn: async () =>
      await getDormitorySharedPosts({
        cardOption,
        filter: debounceFilter,
        search,
        page,
      }).then(res => res),
    enabled,
  });
};

export const useDormitorySharedPost = ({
  postId,
  enabled,
}: {
  postId: number;
  enabled: boolean;
}) =>
  useQuery({
    queryKey: [`/shared/posts/dormitory/${postId}`],
    queryFn: async () =>
      await getDormitorySharedPost(postId).then(response => response.data),
    enabled,
  });

export const useUpdateDormitorySharedPost = () =>
  useMutation<
    AxiosResponse<SuccessBaseDTO>,
    FailureDTO,
    { postId: number; postData: SharedPostProps }
  >({
    mutationFn: updateSharedPost,
  });

export const useDeleteDormitorySharedPost = ({
  postId,
  onSuccess,
  onError,
}: {
  postId: number;
  onSuccess: (data: SuccessBaseDTO) => void;
  onError: (error: Error) => void;
}) =>
  useMutation({
    mutationFn: async () =>
      await deleteDormitorySharedPost(postId).then(response => response.data),
    onSuccess,
    onError,
  });

export const useScrapDormitorySharedPost = () =>
  useMutation<AxiosResponse<SuccessBaseDTO>, FailureDTO, number>({
    mutationFn: scrapDormitoryPost,
  });
