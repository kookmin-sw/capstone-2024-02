'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';

import { CircularButton } from '@/components';
import { UserCard } from '@/components/main-page';
import { useAuthActions, useAuthValue, useUserData } from '@/features/auth';

const styles = {
  container: styled.div`
    width: 100%;
    height: fit-content;
    display: flex;
    flex-direction: column;
    gap: 4.44rem;
  `,
  map: styled.div`
    width: 100%;
    height: 50dvh;
  `,
  mateRecommendationContainer: styled.div`
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 2.6875rem;

    margin-bottom: 5rem;
  `,
  mateRecommendationTitle: styled.h1`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 2rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
    padding: 0 7.75rem;
  `,
  mateRecommendationRow: styled.div`
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    gap: 2.12rem;
    padding: 0 2.5rem;
  `,
  mateRecommendation: styled.div`
    display: flex;
    flex-direction: row;
    gap: 2.625rem;
    overflow-x: auto;

    -ms-overflow-style: none;
    scrollbar-width: none;
    scrollbar ::-webkit-scrollbar {
      display: none;
    }
  `,
};

export function MainPage() {
  const router = useRouter();

  const auth = useAuthValue();
  const { setAuthUserData } = useAuthActions();

  const { data } = useUserData(auth?.accessToken !== undefined);

  const [, setMap] = useState<naver.maps.Map | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);

  const handleScrollRight = () => {
    if (scrollRef.current !== null) {
      scrollRef.current.scrollBy({ left: 300, behavior: 'smooth' });
    }
  };

  const handleScrollLeft = () => {
    if (scrollRef.current !== null) {
      scrollRef.current.scrollBy({ left: -300, behavior: 'smooth' });
    }
  };

  useEffect(() => {
    const center = new naver.maps.LatLng(37.6090857, 126.9966865);
    setMap(
      new naver.maps.Map('map', {
        center,
        disableKineticPan: false,
        scrollWheel: false,
      }),
    );
  }, []);

  useEffect(() => {
    if (data !== undefined) {
      setAuthUserData(data);
      if (data.initialized) {
        // router.replace('/profile');
      }
    }
  }, [data, router, setAuthUserData]);

  return (
    <styles.container>
      <styles.map id="map" />
      <styles.mateRecommendationContainer>
        <styles.mateRecommendationTitle>
          {auth?.user?.name}님의 추천 메이트
        </styles.mateRecommendationTitle>
        <styles.mateRecommendationRow>
          <CircularButton
            direction="left"
            disabled={false}
            onClick={handleScrollLeft}
          />
          <styles.mateRecommendation ref={scrollRef}>
            <Link href="/profile/memberId">
              <UserCard
                name="김마루"
                address="성북 길음동"
                birth={new Date(2000, 5, 27)}
              />
            </Link>
            <Link href="/profile/memberId">
              <UserCard
                name="김마루"
                address="성북 길음동"
                birth={new Date(2000, 5, 27)}
              />
            </Link>
            <Link href="/profile/memberId">
              <UserCard
                name="김마루"
                address="성북 길음동"
                birth={new Date(2000, 5, 27)}
              />
            </Link>
            <Link href="/profile/memberId">
              <UserCard
                name="김마루"
                address="성북 길음동"
                birth={new Date(2000, 5, 27)}
              />
            </Link>
            <Link href="/profile/memberId">
              <UserCard
                name="김마루"
                address="성북 길음동"
                birth={new Date(2000, 5, 27)}
              />
            </Link>
            <Link href="/profile/memberId">
              <UserCard
                name="김마루"
                address="성북 길음동"
                birth={new Date(2000, 5, 27)}
              />
            </Link>
          </styles.mateRecommendation>
          <CircularButton
            direction="right"
            disabled={false}
            onClick={handleScrollRight}
          />
        </styles.mateRecommendationRow>
      </styles.mateRecommendationContainer>
    </styles.container>
  );
}
