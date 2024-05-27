'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';

import { CircularButton } from '@/components';
import { UserCard } from '@/components/main-page';
import { useAuthValue } from '@/features/auth';
import { getGeolocation } from '@/features/geocoding';
import { useRecommendMates } from '@/features/profile';

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

    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;

    p {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1.25rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
    }

    p[class~='caption'] {
      color: #19191980;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 400;
      line-height: normal;
    }
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
    flex-grow: 1;
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
  const auth = useAuthValue();

  const { data: recommendationMates } = useRecommendMates({
    cardOption: 'my',
    enabled: auth?.accessToken != null,
  });

  const [map, setMap] = useState<naver.maps.Map | null>(null);

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
    getGeolocation({
      onSuccess: position => {
        const center = new naver.maps.LatLng(
          position.coords.latitude,
          position.coords.longitude,
        );

        setMap(
          new naver.maps.Map('map', {
            center,
            disableKineticPan: false,
            scrollWheel: false,
          }),
        );
      },
      onError: error => {
        console.error(error);
      },
    });
  }, []);

  return (
    <styles.container>
      <styles.map id="map">
        {map == null && (
          <>
            <p>지도를 불러오는 중입니다.</p>
            <p className="caption">(위치 권한이 필요합니다)</p>
          </>
        )}
      </styles.map>
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
            {recommendationMates?.data?.map(
              ({ memberId, score, nickname, location, profileImageUrl }) => (
                <Link href={`/profile/${memberId}`} key={memberId}>
                  <UserCard
                    name={nickname}
                    percentage={score}
                    profileImage={profileImageUrl}
                    location={location}
                  />
                </Link>
              ),
            )}
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
