'use client';

import styled from 'styled-components';

import { DealTypeFilter } from './filter';
import { SharedPostFilterItem } from './SharedPostFilterItem';

import {
  SharedPostsFilterTypeValue,
  type SharedPostsType,
} from '@/entities/shared-posts-filter';

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
  // TODO: 필터 정보 저장 코드 필요.

  const filterEntries = Object.entries(SharedPostsFilterTypeValue).map<
    [string, string]
  >(([key, value]) => [key, value]);

  const filter = (value: string) => {
    if (selected === 'hasRoom') return true;
    return value === '마이카드' || value === '메이트카드';
  };

  return (
    <styles.container className={className}>
      {filterEntries
        .filter(([, value]) => filter(value))
        .map(([key, value]) => (
          <SharedPostFilterItem key={key} title={value}>
            <DealTypeFilter />
          </SharedPostFilterItem>
        ))}
    </styles.container>
  );
}
