'use client';

import styled from 'styled-components';

import { type SharedPostsType } from '@/entities/shared-posts-filter';
import { useAuthValue, useUserData } from '@/features/auth';

const styles = {
  container: styled.div`
    width: 100%;

    display: flex;
    justify-content: space-evenly;

    @media (max-width: 768px) {
      padding: 0 0.8rem;
    }
  `,
  item: styled.div`
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;

    cursor: pointer;

    padding: 0.75rem 1rem;

    color: var(--Gray-5, #828282);
    font-family: Pretendard;
    font-size: 1.375rem;
    font-style: normal;
    font-weight: 600;
    line-height: 1.5rem;

    border-bottom: 2px solid transparent;

    transition: border-bottom 0.1s ease-in-out;
    &[class~='selected'] {
      color: var(--Main-1, #e15637);
      font-family: Pretendard;
      font-size: 1.375rem;
      font-style: normal;
      font-weight: 700;
      line-height: 1.5rem;

      border-bottom: 2px solid #e15637;

      @media (max-width: 768px) {
        font-size: 1rem;
      }
    }

    @media (max-width: 768px) {
      font-size: 1rem;
      padding: 0.25rem 0.5rem;
    }
  `,
};

interface Props {
  selected: SharedPostsType;
  handleSelect: (selected: SharedPostsType) => void;
}

export function SharedPostsMenu({
  selected,
  handleSelect,
  className,
}: Props & React.ComponentProps<'div'>) {
  const auth = useAuthValue();
  const { data: user } = useUserData(auth?.accessToken != null);

  return (
    <styles.container className={className}>
      <styles.item
        onClick={() => {
          handleSelect('hasRoom');
        }}
        className={selected === 'hasRoom' ? 'selected' : ''}
      >
        방 있는 메이트
      </styles.item>
      <styles.item
        onClick={() => {
          handleSelect('homeless');
        }}
        className={selected === 'homeless' ? 'selected' : ''}
      >
        방 없는 메이트
      </styles.item>
      {user?.univCertified ?? (
        <styles.item
          onClick={() => {
            handleSelect('dormitory');
          }}
          className={selected === 'dormitory' ? 'selected' : ''}
        >
          기숙사 메이트
        </styles.item>
      )}
    </styles.container>
  );
}
