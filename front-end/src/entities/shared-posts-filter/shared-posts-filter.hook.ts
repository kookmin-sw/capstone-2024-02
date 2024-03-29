import { useCallback, useMemo, useState } from 'react';

import { type SharedPostsFilter } from '.';

export const useSharedPostsFilter: () => [
  SharedPostsFilter,
  (
    optionOrUpdater:
      | Partial<SharedPostsFilter>
      | ((prev: SharedPostsFilter) => SharedPostsFilter),
  ) => void,
] = () => {
  const [filter, setFilter] = useState<SharedPostsFilter>({});
  const setter = useCallback(
    (
      optionOrUpdater:
        | Partial<SharedPostsFilter>
        | ((prev: SharedPostsFilter) => SharedPostsFilter),
    ) => {
      if (typeof optionOrUpdater === 'function') {
        setFilter(optionOrUpdater);
      } else {
        setFilter(prev => ({ ...prev, ...optionOrUpdater }));
      }
    },
    [],
  );

  return useMemo(() => [filter, setter], [filter, setter]);
};
