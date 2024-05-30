import { atom } from 'recoil';

import {
  type SelectedExtraOptions,
  type ImageFile,
  type SelectedOptions,
} from './shared.type';

import { type NaverAddress } from '@/features/geocoding';

export const sharedPostPropState = atom<{
  title: string;
  content: string;
  images: ImageFile[];
  mates: Record<
    string,
    { memberId: string; nickname: string; profileImage: string }
  >;
  address?: NaverAddress;
  mateLimit: number;
  expectedMonthlyFee: number;
  houseSize: number;
  selectedExtraOptions: SelectedExtraOptions;
  selectedOptions: SelectedOptions;
  mateCard: {
    gender?: string;
    birthYear?: number;
    location?: string;
    mbti?: string;
    major?: string;
    budget?: string;
    features: {
      smoking?: string;
      roomSharingOption?: string;
      mateAge?: number;
      options: Set<string>;
    };
  };
}>({
  key: 'sharedPostPropState',
  default: {
    title: '',
    content: '',
    images: [],
    mates: {},
    address: undefined,
    mateLimit: 0,
    expectedMonthlyFee: 0,
    houseSize: 0,
    selectedExtraOptions: {},
    selectedOptions: {},
    mateCard: { features: { options: new Set() } },
  },
});
