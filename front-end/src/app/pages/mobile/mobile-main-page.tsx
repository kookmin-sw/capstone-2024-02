'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import styled from 'styled-components';

import { UserCard } from '@/components/main-page';
import { useAuthValue } from '@/features/auth';
import { getGeolocation } from '@/features/geocoding';
import { useRecommendMates } from '@/features/profile';

const styles = {
  container: styled.div`
    display: flex;
    width: 100vw;
    min-width: 390px;
    padding-bottom: 2rem;
    flex-direction: column;
    align-items: center;
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
    gap: 0.5rem;

    margin-bottom: 2.5rem;
  `,
  mateRecommendationTitle: styled.div`
    display: flex;
    padding: 2rem 1.5rem;
    justify-content: flex-start;
    align-items: center;
    align-self: stretch;

    h1 {
      color: #000;

      font-family: 'Noto Sans KR';
      font-size: 1.1875rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
    }
  `,
  mateRecommendationRow: styled.div`
    display: flex;
    width: 100%;
    padding: 0 1.5rem;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
  `,
  mateRecommendation: styled.div`
    display: flex;
    width: 100%;
    padding: 0.5rem 1rem;
    align-items: center;
    gap: 1rem 0.5rem;
    flex-shrink: 0;
    overflow-x: auto;
    flex-wrap: wrap;
  `,
};

export function MobileMainPage() {
  const auth = useAuthValue();

  const { data: recommendationMates } = useRecommendMates({
    cardOption: 'my',
    enabled: auth?.accessToken != null,
  });

  const [map, setMap] = useState<naver.maps.Map | null>(null);

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
          <h1>{auth?.user?.name}님의 추천 메이트</h1>
        </styles.mateRecommendationTitle>
        <styles.mateRecommendation>
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
        </styles.mateRecommendation>
      </styles.mateRecommendationContainer>
    </styles.container>
  );
}
