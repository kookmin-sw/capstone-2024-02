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

const styles = {
  container: styled.div`
    width: 100%;
    display: flex;
    gap: 1rem;

    @media (max-width: 768px) {
      width: 70%;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
  `,
};

export function SharedPostFilters({
  selected,
  className,
}: {
  selected: SharedPostsType;
} & React.ComponentProps<'div'>) {
  const { filter } = useSharedPostsFilter();

  const mateCardFilterTitle = useMemo(() => {
    if (filter.cardType === 'mate') return '메이트 카드';
    if (filter.cardType === 'my') return '마이 카드';
    if (filter.cardType == null) return SharedPostsFilterTypeValue.cardType;
    return 'error';
  }, [filter.cardType]);

  return (
    <styles.container className={className}>
      <SharedPostFilterItem title={mateCardFilterTitle}>
        <MateCardFilter selected={selected} />
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
