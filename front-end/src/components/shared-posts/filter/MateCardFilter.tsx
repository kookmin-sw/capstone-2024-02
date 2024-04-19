'use client';

import styled from 'styled-components';

import { useSharedPostsFilter } from '@/entities/shared-posts-filter';
import { useAuthValue, useUserData } from '@/features/auth';

const styles = {
  container: styled.div`
    display: inline-flex;
    flex-direction: column;
    width: fit-content;
    gap: 1rem;

    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;

    button {
      all: unset;

      cursor: pointer;
      padding: 1rem;
      transition: 200ms background-color ease-in-out;
      border-radius: 8px;
      min-width: 15rem;
    }

    button:hover {
      background-color: #ededed;
    }
  `,
};

export function MateCardFilter() {
  const auth = useAuthValue();
  const { data: userData } = useUserData(auth?.accessToken != null);

  const { setFilter } = useSharedPostsFilter();

  return (
    <styles.container>
      <button
        type="button"
        onClick={() => {
          setFilter(prev => ({ ...prev, cardType: 'mate' }));
        }}
      >
        {userData?.name}님이 원하는 방 찾기
      </button>
      <button
        type="button"
        onClick={() => {
          setFilter(prev => ({ ...prev, cardType: 'my' }));
        }}
      >
        {userData?.name}님을 원하는 방 찾기
      </button>
    </styles.container>
  );
}
