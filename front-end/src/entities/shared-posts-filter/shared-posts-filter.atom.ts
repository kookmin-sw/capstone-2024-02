import { atom } from 'recoil';

import { type SharedPostsFilter } from './shared-posts-filter.type';

export const sharedPostsFilterState = atom<SharedPostsFilter>({
  key: 'sharedPostsFilterState',
  default: {
    roomInfo: {
      hasLivingRoom: false,
    },
    extraInfo: {},
  },
});
