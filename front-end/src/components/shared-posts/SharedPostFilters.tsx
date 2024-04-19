'use client';

import { useMemo } from 'react';
import styled from 'styled-components';

import {
  DealTypeFilter,
  ExtraInfoFilter,
  MateCardFilter,
  RoomTypeFilter,
} from './filter';
import { SharedPostFilterItem } from './SharedPostFilterItem';

import {
  SharedPostsFilterTypeValue,
  useSharedPostsFilter,
  type SharedPostsType,
} from '@/entities/shared-posts-filter';
import { useAuthValue, useUserData } from '@/features/auth';

const styles = {
  container: styled.div`
    display: flex;
    gap: 1rem;
  `,
};

export function SharedPostFilters({
  selected,
  className,
}: {
  selected: SharedPostsType;
} & React.ComponentProps<'div'>) {
  const auth = useAuthValue();
  const { data: userData } = useUserData(auth?.accessToken != null);

  const { filter } = useSharedPostsFilter();

  const mateCardFilterTitle = useMemo(() => {
    if (filter.cardType == null) return SharedPostsFilterTypeValue.cardType;
    if (filter.cardType === 'mate') return `${userData?.name}님이 원하는 방`;
    if (filter.cardType === 'my') return `${userData?.name}님을 원하는 방`;
    return 'error';
  }, [userData?.name, filter.cardType]);

  return (
    <styles.container className={className}>
      <SharedPostFilterItem title={mateCardFilterTitle}>
        <MateCardFilter />
      </SharedPostFilterItem>
      {selected === 'hasRoom' && (
        <>
          <SharedPostFilterItem title={SharedPostsFilterTypeValue.dealInfo}>
            <DealTypeFilter />
          </SharedPostFilterItem>
          <SharedPostFilterItem title={SharedPostsFilterTypeValue.roomInfo}>
            <RoomTypeFilter />
          </SharedPostFilterItem>
          <SharedPostFilterItem title={SharedPostsFilterTypeValue.extraInfo}>
            <ExtraInfoFilter />
          </SharedPostFilterItem>
        </>
      )}
    </styles.container>
  );
}
