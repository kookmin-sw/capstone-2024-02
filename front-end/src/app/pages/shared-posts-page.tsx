'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';

import { CircularButton } from '@/components';
import { UserCard } from '@/components/main-page';
import {
  PostCard,
  SharedPostFilters,
  SharedPostsMenu,
} from '@/components/shared-posts';
import {
  useSharedPostsFilter,
  type SharedPostsType,
} from '@/entities/shared-posts-filter';
import { useAuthValue } from '@/features/auth';
import { useRecommendMates } from '@/features/profile';
import {
  useDormitorySharedPosts,
  usePaging,
  useSharedPosts,
  type GetDormitorySharedPostsDTO,
  type GetSharedPostsDTO,
} from '@/features/shared';

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
  createButtonRow: styled.div`
    display: flex;
    align-items: center;
    justify-content: space-between;
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
    gap: 0.25rem;

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
  const [prevSharedPosts, setPrevSharedPosts] = useState<
    GetSharedPostsDTO | GetDormitorySharedPostsDTO | null
  >(null);

  const { filter, derivedFilter, reset: resetFilter } = useSharedPostsFilter();

  const {
    page,
    sliceSize,
    currentSlice,
    isFirstPage,
    isLastPage,
    handleSetPage,
    handleNextPage,
    handlePrevPage,
  } = usePaging({
    totalPages: totalPageCount,
    sliceSize: 10,
  });

  const { data: sharedPosts } = useSharedPosts({
    filter: derivedFilter,
    cardOption: filter.cardType ?? 'my',
    enabled: auth?.accessToken != null && selected === 'hasRoom',
    page: page - 1,
  });

  const { data: dormitorySharedPosts } = useDormitorySharedPosts({
    filter: derivedFilter,
    cardOption: filter.cardType ?? 'my',
    enabled: auth?.accessToken != null && selected === 'dormitory',
    page: page - 1,
  });

  const posts = useMemo(
    () => (selected === 'hasRoom' ? sharedPosts : dormitorySharedPosts),
    [selected, sharedPosts, dormitorySharedPosts],
  );

  const { data: recommendationMates } = useRecommendMates({
    enabled: auth?.accessToken != null && selected === 'homeless',
    cardOption: filter.cardType ?? 'my',
  });

  useEffect(() => {
    resetFilter();
    return () => {
      resetFilter();
    };
  }, [selected, resetFilter]);

  useEffect(() => {
    if (selected === 'hasRoom' && sharedPosts != null) {
      setTotalPageCount(sharedPosts.data.totalPages);
      setPrevSharedPosts(null);
    } else if (selected === 'dormitory' && dormitorySharedPosts != null) {
      setTotalPageCount(dormitorySharedPosts.data.totalPages);
      setPrevSharedPosts(null);
    }
  }, [selected, dormitorySharedPosts, sharedPosts]);

  return (
    <styles.container>
      <styles.SharedPostsMenu selected={selected} handleSelect={setSelected} />
      <styles.createButtonRow>
        <SharedPostFilters selected={selected} />
        {(selected === 'hasRoom' || selected === 'dormitory') && (
          <styles.createButton
            onClick={() => {
              router.push(
                `/shared/writing/${selected === 'hasRoom' ? 'room' : 'dormitory'}`,
              );
            }}
          >
            작성하기
          </styles.createButton>
        )}
      </styles.createButtonRow>
      {selected === 'hasRoom' || selected === 'dormitory' ? (
        <>
          <styles.posts>
            {prevSharedPosts != null
              ? prevSharedPosts.data.content.map(post => (
                  <PostCard
                    key={post.id}
                    post={post}
                    onClick={() => {
                      router.push(`/shared/${post.id}`);
                    }}
                  />
                ))
              : posts?.data.content.map(post => (
                  <PostCard
                    key={post.id}
                    post={post}
                    onClick={() => {
                      router.push(
                        `/shared/${selected === 'hasRoom' ? 'room' : 'dormitory'}/${post.id}`,
                      );
                    }}
                  />
                ))}
          </styles.posts>
          {posts != null && posts.data.content.length !== 0 && (
            <styles.pagingRow>
              <styles.CircularButton
                direction="left"
                disabled={isFirstPage}
                onClick={() => {
                  if (sharedPosts != null) {
                    setPrevSharedPosts(sharedPosts);
                  }
                  handlePrevPage();
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
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
                    onClick={() => {
                      if (sharedPosts != null) {
                        setPrevSharedPosts(sharedPosts);
                      }
                      handleSetPage(index + 1 + currentSlice * sliceSize);
                      window.scrollTo({ top: 0, behavior: 'smooth' });
                    }}
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
                onClick={() => {
                  if (sharedPosts != null) {
                    setPrevSharedPosts(sharedPosts);
                  }
                  handleNextPage();
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
              />
            </styles.pagingRow>
          )}
        </>
      ) : (
        <styles.cards>
          {recommendationMates?.data?.map(
            ({ memberId, score, nickname, location, profileImageUrl }) => (
              <Link href={`/profile/${memberId}`} key={memberId}>
                <UserCard
                  name={nickname}
                  percentage={score}
                  location={location}
                  profileImage={profileImageUrl}
                />
              </Link>
            ),
          )}
        </styles.cards>
      )}
    </styles.container>
  );
}
