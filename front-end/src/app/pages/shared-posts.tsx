'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import styled from 'styled-components';

import { CircularButton } from '@/components';
import { UserCard } from '@/components/main-page';
import {
  SharedPostsMenu,
  SharedPostFilters,
  PostCard,
} from '@/components/shared-posts';
import { type SharedPostsType } from '@/entities/shared-posts-filter';
import { useAuthActions, useAuthValue, useUserData } from '@/features/auth';
import { usePaging, useSharedPosts } from '@/features/shared';

const styles = {
  container: styled.div`
    padding-top: 4.12rem;
    padding-inline: 16rem;
    width: 100%;
    height: fit-content;

    display: flex;
    flex-direction: column;
  `,
  SharedPostsMenu: styled(SharedPostsMenu)`
    margin-bottom: 2rem;
  `,
  SharedPostsFilter: styled(SharedPostFilters)`
    margin-bottom: 2.81rem;
  `,
  createButtonRow: styled.div`
    display: flex;
    justify-content: end;
    align-items: center;
    margin-bottom: 4.12rem;
  `,
  createButton: styled.button`
    all: unset;

    cursor: pointer;

    display: flex;
    width: 7.125rem;
    padding: 0.5rem 1.5rem;
    justify-content: center;
    align-items: center;

    border-radius: 8px;
    background: var(--Black, #35373a);

    color: #fff;
    font-family: Pretendard;
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 600;
    line-height: 1.5rem;
  `,
  posts: styled.div`
    display: flex;
    flex-direction: column;
    padding-inline: 2rem;
    gap: 2rem;
  `,
  CircularButton: styled(CircularButton)`
    scale: 0.9;
  `,
  pagingRow: styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-block: 7rem;
  `,
  paging: styled.div`
    display: flex;
    gap: 3rem;

    button {
      all: unset;
      cursor: pointer;

      color: #b2b2b2;
      font-family: 'Spoqa Han Sans Neo';
      font-size: 1.25rem;
      font-style: normal;
      font-weight: 400;
      line-height: 120%;

      &[class~='current'] {
        color: var(--Black, #35373a);
        font-family: 'Spoqa Han Sans Neo';
        font-size: 1.25rem;
        font-style: normal;
        font-weight: 700;
        line-height: 120%;
        text-decoration-line: underline;
      }
    }
  `,
  cards: styled.div`
    padding-left: 2.62rem;
    display: flex;
    flex-wrap: wrap;
    gap: 2rem 2.62rem;
  `,
};

export function SharedPostsPage() {
  const router = useRouter();

  const auth = useAuthValue();
  const [selected, setSelected] = useState<SharedPostsType>('hasRoom');
  const [totalPageCount, setTotalPageCount] = useState(0);
  const { setAuthUserData } = useAuthActions();

  const { data: userData } = useUserData(auth?.accessToken !== undefined);

  const {
    page,
    sliceSize,
    currentSlice,
    isFirstPage,
    isLastPage,
    handleNextPage,
    handlePrevPage,
  } = usePaging({
    totalPages: totalPageCount,
    sliceSize: 10,
  });

  const { data: sharedPosts } = useSharedPosts({
    enabled: auth?.accessToken !== undefined && selected === 'hasRoom',
    page: page - 1,
  });

  useEffect(() => {
    if (sharedPosts !== undefined) {
      setTotalPageCount(sharedPosts.data.totalPages);
    }
  }, [sharedPosts]);

  useEffect(() => {
    if (userData !== undefined) {
      setAuthUserData(userData);
      if (userData.initialized) {
        // router.replace('/profile');
      }
    }
  }, [userData, router, setAuthUserData]);

  return (
    <styles.container>
      <styles.SharedPostsMenu selected={selected} handleSelect={setSelected} />
      <styles.SharedPostsFilter selected={selected} />
      {selected === 'hasRoom' ? (
        <>
          <styles.createButtonRow>
            <Link href="/shared/writing">
              <styles.createButton>작성하기</styles.createButton>
            </Link>
          </styles.createButtonRow>
          <styles.posts>
            {sharedPosts?.data.content.map(post => (
              <Link key={post.id} href={`/shared/${post.id}`}>
                <PostCard />
              </Link>
            ))}
          </styles.posts>
          <styles.pagingRow>
            <styles.CircularButton
              direction="left"
              disabled={isFirstPage}
              onClick={handlePrevPage}
            />
            <styles.paging>
              {Array.from({
                length: Math.min(
                  totalPageCount - currentSlice * sliceSize,
                  sliceSize,
                ),
              }).map((_, index) => (
                <button
                  type="button"
                  // eslint-disable-next-line react/no-array-index-key
                  key={`${currentSlice}-${index}`}
                  className={
                    page === index + 1 + currentSlice * sliceSize
                      ? 'current'
                      : ''
                  }
                >
                  {index + 1 + currentSlice * sliceSize}
                </button>
              ))}
            </styles.paging>
            <styles.CircularButton
              direction="right"
              disabled={isLastPage}
              onClick={handleNextPage}
            />
          </styles.pagingRow>
        </>
      ) : (
        <styles.cards>
          <Link href="/profile/memberId">
            <UserCard name="" address="" birth={new Date(2000, 5, 27)} />
          </Link>
          <Link href="/profile/memberId">
            <UserCard name="" address="" birth={new Date(2000, 5, 27)} />
          </Link>
          <Link href="/profile/memberId">
            <UserCard name="" address="" birth={new Date(2000, 5, 27)} />
          </Link>
          <Link href="/profile/memberId">
            <UserCard name="" address="" birth={new Date(2000, 5, 27)} />
          </Link>
          <Link href="/profile/memberId">
            <UserCard name="" address="" birth={new Date(2000, 5, 27)} />
          </Link>
          <Link href="/profile/memberId">
            <UserCard name="" address="" birth={new Date(2000, 5, 27)} />
          </Link>
        </styles.cards>
      )}
    </styles.container>
  );
}
