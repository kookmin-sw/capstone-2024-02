import { useMemo } from 'react';
import { useRecoilState, useResetRecoilState } from 'recoil';

import { sharedPostsFilterState } from './shared-posts-filter.atom';
import { type SharedPostsFilter } from './shared-posts-filter.type';

export const useSharedPostsFilter = () => {
  const [filter, setFilter] = useRecoilState<SharedPostsFilter>(
    sharedPostsFilterState,
  );

  const reset = useResetRecoilState(sharedPostsFilterState);
  return useMemo(
    () => ({
      filter,
      setFilter,
      reset,
    }),
    [filter, setFilter, reset],
  );
};
