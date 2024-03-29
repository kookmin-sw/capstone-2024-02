'use client';

import { type HTMLAttributes } from 'react';
import styled from 'styled-components';

import { DropDownList } from './DropDownList';

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

export function SharedPostsFilter({
  selected,
  className,
}: {
  selected: SharedPostsType;
} & HTMLAttributes<HTMLDivElement>) {
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
          <DropDownList
            key={key}
            title={value}
            items={[]}
            onSelect={() => {}}
          />
        ))}
    </styles.container>
  );
}
