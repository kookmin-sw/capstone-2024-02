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
} from '@/features/shared';

const styles = {
  container: styled.div`
    display: flex;
    align-items: center;
    width: 100vw;
    min-width: 390px;
    flex-direction: column;
    gap: 1rem;
    padding: 2rem 1rem;
  `,
  SharedPostsMenu: styled(SharedPostsMenu)`
    margin-bottom: 1rem;
  `,
  createButtonRow: styled.div`
    display: flex;
    width: 100%;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;

    @media (max-width: 768px) {
      align-items: flex-start;
    }
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

    @media (max-width: 768px) {
      font-size: 0.75rem;
      width: 4rem;
      padding: 0.25rem 1rem;
    }
  `,
  posts: styled.div`
    display: flex;
    flex-direction: column;
    padding-inline: 1rem;
    gap: 1rem;

    @media (max-width: 768px) {
      width: 100%;
    }
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

    @media (max-width: 768px) {
      gap: 1rem 1rem;
      padding-left: 0;
      justify-content: flex-start;
    }
  `,
  noRecommendation: styled.div`
    font-family: 'Noto Sans KR';
    font-size: 0.85rem;
    font-style: normal;
    font-weight: 500;

    display: flex;
    width: 100%;
    justify-content: center;
  `,
};

export function MobileSharedPostsPage() {
  const router = useRouter();

  const auth = useAuthValue();
  const [selected, setSelected] = useState<SharedPostsType>('hasRoom');
  const [totalPageCount, setTotalPageCount] = useState(0);

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
      setTotalPageCount(sharedPosts.totalPages);
    } else if (selected === 'dormitory' && dormitorySharedPosts != null) {
      setTotalPageCount(dormitorySharedPosts.totalPages);
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
            {posts != null && posts.data.length > 0 ? (
              posts.data.map(post => (
                <PostCard
                  key={post.id}
                  post={post}
                  onClick={() => {
                    router.push(
                      `/shared/${selected === 'hasRoom' ? 'room' : 'dormitory'}/${post.id}`,
                    );
                  }}
                />
              ))
            ) : (
              <styles.noRecommendation>
                <p>추천되는 게시글이 없습니다.</p>
              </styles.noRecommendation>
            )}
          </styles.posts>
          {posts?.data.length !== 0 && (
            <styles.pagingRow>
              <styles.CircularButton
                direction="left"
                disabled={isFirstPage}
                onClick={() => {
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
                  handleNextPage();
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
              />
            </styles.pagingRow>
          )}
        </>
      ) : (
        <styles.cards>
          {recommendationMates != null && recommendationMates.length > 0 ? (
            recommendationMates.map(
              ({
                memberId,
                score,
                nickname,
                location,
                profileImageUrl,
                options,
              }) => (
                <Link href={`/profile/${memberId}`} key={memberId}>
                  <UserCard
                    name={nickname}
                    percentage={score}
                    profileImage={profileImageUrl}
                    location={location}
                    smoking={options.smoking}
                    roomSharingOption={options.roomSharingOption}
                    mateAge={options.mateAge}
                    hideScore={false}
                  />
                </Link>
              ),
            )
          ) : (
            <styles.noRecommendation>
              <p>추천되는 메이트가 없습니다.</p>
            </styles.noRecommendation>
          )}
        </styles.cards>
      )}
    </styles.container>
  );
}
