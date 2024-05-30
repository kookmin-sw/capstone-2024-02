'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import styled from 'styled-components';

import { CircularButton, CircularProfileImage } from '@/components';
import { UserCard } from '@/components/main-page';
import { useAuthValue } from '@/features/auth';
import { fromAddrToCoord, getGeolocation } from '@/features/geocoding';
import { useRecommendMates } from '@/features/profile';
import { ColorRing } from 'react-loader-spinner';

const styles = {
  container: styled.div`
    width: 100%;
    height: fit-content;
    display: flex;
    flex-direction: column;
    gap: 3rem;
    padding: 0rem 0rem 2rem 0rem;
  `,
  map: styled.div`
    width: 100%;
    height: 30rem;

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
    gap: 3rem;
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
    gap: 2rem;
    padding: 0 2.25rem;
  `,
  mateRecommendation: styled.div`
    display: flex;
    flex-grow: 1;
    flex-direction: row;
    gap: 2rem;
    overflow-x: auto;

    -ms-overflow-style: none;
    scrollbar-width: none;
    scrollbar ::-webkit-scrollbar {
      display: none;
    }
  `,
  mateRecommendationIsEmpty: styled.div`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1.5rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    display: flex;
    width: 100%;
    justify-content: center;
    padding-block: 2rem;
  `,
};

function Marker({
  nickname,
  profileImage,
  score,
}: {
  score: number;
  profileImage: string;
  nickname: string;
}) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '0.75rem',
        scale: 0.75,
      }}
    >
      <CircularProfileImage
        diameter={100}
        percentage={score}
        url={profileImage}
        hideScore={false}
      />
      <p>{nickname}</p>
    </div>
  );
}

export function MainPage() {
  const auth = useAuthValue();

  const { data: recommendationMates } = useRecommendMates({
    cardOption: 'my',
    enabled: auth?.accessToken != null,
  });

  const [map, setMap] = useState<naver.maps.Map | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);

  const router = useRouter();

  const [permission, setPermission] = useState<string>();

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
    (async () => {
      try {
        const permissionStatus = await navigator.permissions.query({
          name: 'geolocation',
        });

        permissionStatus.onchange = () => {
          setPermission(permissionStatus.state);
        };
      } catch (error) {
        console.error('Error checking geolocation permission:', error);
      }
    })();
  }, []);

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

  const [createdMarkers, setCreatedMarkers] = useState<naver.maps.Marker[]>([]);

  useEffect(() => {
    if (map == null || recommendationMates == null) return;

    recommendationMates.forEach(mate => {
      fromAddrToCoord({ query: mate.location }).then(res => {
        const address = res.shift();
        if (address == null) return;

        const spot = new naver.maps.LatLng(+address.y, +address.x);
        const marker = new naver.maps.Marker({
          position: spot,
          map,
          icon: {
            content: renderToStaticMarkup(
              <Marker
                nickname={mate.nickname}
                profileImage={mate.profileImageUrl}
                score={mate.score}
              />,
            ),
          },
        });

        marker.addListener('click', () => {
          router.push(`/profile/${mate.memberId}`);
        });

        setCreatedMarkers(prev => prev.concat(marker));
      });
    });
  }, [map, recommendationMates, router]);

  useEffect(() => {
    if (map == null) return () => {};

    const showMarker = (
      targetMap: naver.maps.Map,
      marker: naver.maps.Marker,
    ) => {
      if (marker.getMap() != null) return;
      marker.setMap(targetMap);
    };

    const hideMarker = (marker: naver.maps.Marker) => {
      if (marker.getMap() == null) return;
      marker.setMap(null);
    };

    const updateMarkers = (
      targetMap: naver.maps.Map | null,
      markers: naver.maps.Marker[],
    ) => {
      if (targetMap == null) return;

      const mapBounds = targetMap.getBounds();
      let marker: naver.maps.Marker, position;

      for (let i = 0; i < markers.length; i += 1) {
        marker = markers[i];
        position = marker.getPosition();

        if (mapBounds.hasPoint(position)) {
          showMarker(targetMap, marker);
        } else {
          hideMarker(marker);
        }
      }
    };

    const MoveEventListner = naver.maps.Event.addListener(map, 'idle', () => {
      updateMarkers(map, createdMarkers);
    });
    return () => {
      naver.maps.Event.removeListener(MoveEventListner);
    };
  }, [createdMarkers, map]);

  const renderIndicator = () => {
    if (permission != null && permission !== 'granted')
      return <p className="caption">(위치 권한이 필요합니다)</p>;
    if (map == null)
      return (
        <ColorRing
          visible={true}
          height="80"
          width="80"
          ariaLabel="color-ring-loading"
          wrapperStyle={{}}
          wrapperClass="color-ring-wrapper"
          colors={['#e15b64', '#f47e60', '#f8b26a', '#abbd81', '#849b87']}
        />
      );
    return <></>;
  };

  return (
    <styles.container>
      <styles.map id="map">{renderIndicator()}</styles.map>
      <styles.mateRecommendationContainer>
        <styles.mateRecommendationTitle>
          {auth?.user?.name}님의 추천 메이트
        </styles.mateRecommendationTitle>
        <styles.mateRecommendationRow>
          {recommendationMates != null && recommendationMates.length > 0 ? (
            <>
              <CircularButton
                direction="left"
                disabled={false}
                onClick={handleScrollLeft}
              />
              <styles.mateRecommendation ref={scrollRef}>
                {recommendationMates.map(
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
                )}
              </styles.mateRecommendation>
              <CircularButton
                direction="right"
                disabled={false}
                onClick={handleScrollRight}
              />
            </>
          ) : (
            <styles.mateRecommendationIsEmpty style={{ alignSelf: 'center' }}>
              <p>추천되는 메이트가 없습니다.</p>
            </styles.mateRecommendationIsEmpty>
          )}
        </styles.mateRecommendationRow>
      </styles.mateRecommendationContainer>
    </styles.container>
  );
}
