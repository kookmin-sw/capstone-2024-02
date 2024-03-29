'use client';

import { useEffect, useState } from 'react';
import styled from 'styled-components';

import {
  Bookmark,
  CircularProfileImage,
  HorizontalDivider,
} from '@/components';
import {
  CardToggleButton,
  ImageGrid,
  MiniCircularProfileImage,
} from '@/components/shared-post-page';
import { type SharedPost } from '@/entities/shared-post';

const styles = {
  container: styled.div`
    background: var(--background, #f7f6f9);

    position: relative;
    left: -15rem;
    width: 100dvw;
    min-height: 100%;
    height: fit-content;

    display: flex;
    justify-content: center;
    align-items: stretch;
    padding: 3rem 0;
    gap: 1rem;

    overflow: auto;
  `,
  houseInfo: styled.div`
    display: flex;
    flex-direction: column;
    gap: 1.5rem;

    width: 50%;
    min-height: 80dvh;
    height: 100%;
    border-radius: 16px;
    background: #fff;
    padding: 1.5rem;
  `,
  hostInfo: styled.div`
    display: flex;
    flex-direction: column;

    position: relative;

    width: 100%;
    height: 100%;
    border-radius: 16px;
    background: #fff;

    padding: 2rem 3.5rem;
  `,
  titleRow: styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-right: 1.125rem;
  `,
  title: styled.h1`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1.5rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
  briefInfoContainer: styled.div`
    display: flex;
    align-items: end;
    justify-content: space-between;

    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    div {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
  `,
  viewCount: styled.p`
    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
    margin-right: 1rem;
  `,
  content: styled.div`
    display: flex;
    flex-direction: column;
    gap: 2rem;
  `,
  dealInfo: styled.div`
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
  `,
  dealTitle: styled.h2`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;

    margin-bottom: 1.25rem;
  `,
  dealContent: styled.div`
    display: flex;
    flex-direction: column;
    gap: 2rem;
    margin: 0 0.63rem;
  `,
  dealItemContainer: styled.div`
    display: flex;
    justify-content: space-between;

    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
  `,
  roomInfo: styled.div`
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
  `,
  roomInfoTitle: styled.h2`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;

    margin-bottom: 1.25rem;
  `,
  roomInfoContent: styled.div`
    display: flex;
    flex-direction: column;
    gap: 2rem;
    margin: 0 0.63rem;
  `,
  detailInfoContanier: styled.div`
    display: flex;
    flex-direction: column;
  `,
  detailInfoTitle: styled.h2`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
    margin-bottom: 1.25rem;
  `,
  detailInfoContent: styled.p`
    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
  `,
  locationInfoContainer: styled.div`
    display: flex;
    flex-direction: column;
    gap: 1.31rem;
  `,
  locationInfoTitle: styled.h2`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
  locationInfoContent: styled.p`
    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;

    cursor: pointer;
  `,
  map: styled.div`
    width: 100%;
    height: 21.125rem;
  `,
  locationEnvInfo: styled.div`
    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;

    display: flex;
    flex-direction: column;
    gap: 2rem;
  `,
  hostInfoContainer: styled.div`
    display: flex;
    align-items: center;
    gap: 3.125rem;
    margin-bottom: 3.125rem;
  `,
  roomParticipantsContainer: styled.div`
    display: flex;
    flex-direction: column;
    gap: 1.5rem;

    width: 25%;
    min-height: 50dvh;
    height: 50%;
  `,
  roomParticipants: styled.div`
    display: flex;

    width: 100%;
    height: 6.125rem;
    flex-shrink: 0;
    border-radius: 16px;
    background: #fff;

    padding: 0.44rem 1.56rem;
    overflow-x: auto;
  `,
  hostInfoContent: styled.div`
    display: flex;
    flex-direction: column;

    h1 {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1.5rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;
      margin-bottom: 1rem;
    }

    p {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;
      margin-bottom: 0.25rem;
    }
  `,
  hostCardContent: styled.div`
    display: flex;
    flex-direction: column;
    gap: 0.5rem;

    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    margin-bottom: 3.12rem;

    .essential {
      color: #e15637;
    }
  `,
  hostButtonsContainer: styled.div`
    display: flex;
    flex-direction: column;
    gap: 1rem;

    div {
      display: flex;
      gap: 1rem;
    }

    button {
      all: unset;
      cursor: pointer;

      flex-grow: 1;

      display: flex;
      padding: 0.5rem 1.5rem;
      justify-content: center;
      align-items: center;

      border-radius: 8px;
      border: 1px solid var(--Gray-3, #888);
      background: var(--White, #fff);

      color: var(--Gray-3, #888);
      font-family: Pretendard;
      font-size: 1.125rem;
      font-style: normal;
      font-weight: 600;
      line-height: 1.5rem;
    }

    .color {
      background: var(--Main-1, #35373a);
      border: 1px solid var(--Main-1);
      color: #fff;
    }

    margin-bottom: 2.69rem;
  `,
  cardToggleButtonContainer: styled.div`
    display: flex;
    position: relative;
    flex-direction: column;
    gap: 1.06rem;

    left: -2.25rem;

    width: calc(100% + 2.25rem);
  `,
};

const dummyImages = [
  'https://s3-alpha-sig.figma.com/img/efd0/12b5/6a0078a4aa75b0e9a9fb53a6d9a7c560?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=B4Aqn-hPJ0ihHPDfRzMOJKsIyqx03layytoxWWfcKWjb5Y99kfmk79oYrVGeBG1r9Upf2S4T9qEoYombOIKwE5BQ-uHa5bHjqSrrGN-Ki4V9uwv-FoINnD1-yK8DFE75XlMlQAWIw7c19End47NjTtAF7aZwN~votmj7HtmRK5~ftLRDqNL~SuLN0-LoMGpuDPhN2DMbafK1KOZeP5ur-dZKNfLQlg7KoOk8DcfS6p1IkoLY56tYQgrDmNG9YzXPSRsjrXAO1BDvYQRw0xdepU9cc63~lYlFbQu0j15PcUKGt9BjkjLwxB5sibmX6qG4Ct4tu161kpw3~urASb84eg__',
  'https://s3-alpha-sig.figma.com/img/ff85/788d/96b4a3ec1b31b6baf36b11c772529753?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=cjUTegC0Ory6cMHPNzEqOS8K3anjn8DBCi~2j6hiEcZS1exKyuo3g0jfke26Y~HVjFa0YhUjW8tjxjd7KBmgLWHxow2U8YwP6bNJBHeNgKMwpDHk2aF1vUPO-nfDEM7ZCfZCSsWcUz~eNS5XP5XwFvd7aP8g~62hJnvdtkdMpP3qiCS70w~QUAdwBtb57TjBD3EWelFbKg2JcWTVlIxYd7qtj~uuP8-0HXrgmdssska9T3tCF0aT6REa4QFa7dYggiKL3rk8aLyjJ46RbOm9l~mqbIs9PqQakObhU~621G1PeAX9ZtkgzryhtNWG6EKyGTzhEvm4~-~AWriZFAVebw__',
  'https://s3-alpha-sig.figma.com/img/0de8/113e/7a56e61d0a97703ddef68f41508a60f9?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=SMYm-FhubQ80cjTUJD3g2DJc17u7aOu8~vo7FDtQpmPh4yrJen0RmaxnLXKRLoz1NhcLEG~x65veb7hZcZBgjf5m~ayGDhEwKE4P7gKWPYh3uPT8vW1aGQsAi0JB~fw7qA7nlaVCpPANPIaQLo9LgiRvCEuTRaxxFgwSBXcHxFbGhZCDmUzKvSJgk6Z5TE6jDqOj2nDSDRGu2U0keA5nJu8nvyD4HwJGRDLfNyVDFMa19HDaT2Ma34wVhOSClS-7jUhr4UYwjgLVostmHaFH9OcTtgRWw~UHXSsW30Yrjao2YsDVbskurQ81cjr7SusT-RRXJUQoL1sT91utNz4oBw__',
  'https://s3-alpha-sig.figma.com/img/2d52/bfda/0a900d41e2a6d77dd731fac06577540c?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=l0b7Up505QbH4ZH5Z0vATZlKNVq70rhtoZVCeabC7RT6kc9q-WQ5gGl4S2~s0JDCaB4lhVuvQMm3m0FxzvjTTmcW0il7~U1ozz2sdgOaxMpX35~0C0lS1nkVlnH6Karq4XQLCOVU1So~bjORg2CkS3mlXNbReSsNKgb-V594MzT25yZmaQ6Gnv3tS7f0pLH-QHRBvwqZRASrJ1ud8K2b1J-JD5KwUneP-gGi8-BKX36Is6VHdkAD1RwJnasfDee540UUCJzrxQAC4ODGFB8cnD0Mv4UsF4vtOncskz77EjclHb-FawpF2lLBYwsZDnSv21dGeF328LeNNS64a-wSCQ__',
  'https://s3-alpha-sig.figma.com/img/2006/8205/5f945443386acf4bfb5a6853b361442f?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=W8SbDFzsK1gVBUcE~OxQdVW5EI7fRPGiBeGf7loAp73pnO4gVffx7xuTeMQB7CiTd~2mlRDws4vB-5-eQEzrqk4W2lCwAC27tyEMJ-Oh55Pzo0KGDQhQgcbGZ~FRlj9vvgvVK-4cZiK3UrBWgrYN8eSXb~3sp7HQBDY~U0YBCZh9W6bcqlkjZUhV1YIRjdDy~d-b81FvYAI-GIU5JiU5SC0-IGNU5X3l33mcgbADRaC4mdV~uKSCGpOGtDEXOdwTtAODlW2aY28-JAM84hOl6YFXmjjVJlHk13YPcnb~KWo-bX0DOYBnY3Ad3rf0BdgkYjL2ejUS6DE2l613agQKcA__',
];

const dummyParticipants = [
  'https://s3-alpha-sig.figma.com/img/59a5/3c6f/ae49249b51c7d5d81ab89eeb0bf610f1?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=bS1lS~7-WQsD9x5vHJBOiMnhhFjI~VgCwJH6Bzz~IxFWob-PV-XZweWFIhU6yJC3XHv5qZKZxnP9RWT~0ciIbQfofuhbODEUxnMHe6oq8Dl45khsD30dnXOK~FLBPpWhMumJO-zPpuWjiRwsZ35mfWLbgyT7dND41I9yXCyRASQx9v2iAGzDoVzTfvtkjRyGw6es6fSXRsFGMqthnzYmv7DZT~FCz2avi3-NqXruXQpkijQHNEQUM61ThFiNYEIv8vb1wZWf-USbbJpE8bdbUneblY2T0cWwMRBtKbCrJ0Y~P9lvFbzqBv7h9WOzNyJW~~KeG9vVrBmLRRo1BsNdng__',
  'https://s3-alpha-sig.figma.com/img/2d52/bfda/0a900d41e2a6d77dd731fac06577540c?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=l0b7Up505QbH4ZH5Z0vATZlKNVq70rhtoZVCeabC7RT6kc9q-WQ5gGl4S2~s0JDCaB4lhVuvQMm3m0FxzvjTTmcW0il7~U1ozz2sdgOaxMpX35~0C0lS1nkVlnH6Karq4XQLCOVU1So~bjORg2CkS3mlXNbReSsNKgb-V594MzT25yZmaQ6Gnv3tS7f0pLH-QHRBvwqZRASrJ1ud8K2b1J-JD5KwUneP-gGi8-BKX36Is6VHdkAD1RwJnasfDee540UUCJzrxQAC4ODGFB8cnD0Mv4UsF4vtOncskz77EjclHb-FawpF2lLBYwsZDnSv21dGeF328LeNNS64a-wSCQ__',
  'https://s3-alpha-sig.figma.com/img/2006/8205/5f945443386acf4bfb5a6853b361442f?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=W8SbDFzsK1gVBUcE~OxQdVW5EI7fRPGiBeGf7loAp73pnO4gVffx7xuTeMQB7CiTd~2mlRDws4vB-5-eQEzrqk4W2lCwAC27tyEMJ-Oh55Pzo0KGDQhQgcbGZ~FRlj9vvgvVK-4cZiK3UrBWgrYN8eSXb~3sp7HQBDY~U0YBCZh9W6bcqlkjZUhV1YIRjdDy~d-b81FvYAI-GIU5JiU5SC0-IGNU5X3l33mcgbADRaC4mdV~uKSCGpOGtDEXOdwTtAODlW2aY28-JAM84hOl6YFXmjjVJlHk13YPcnb~KWo-bX0DOYBnY3Ad3rf0BdgkYjL2ejUS6DE2l613agQKcA__',
  'https://s3-alpha-sig.figma.com/img/0de8/113e/7a56e61d0a97703ddef68f41508a60f9?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=SMYm-FhubQ80cjTUJD3g2DJc17u7aOu8~vo7FDtQpmPh4yrJen0RmaxnLXKRLoz1NhcLEG~x65veb7hZcZBgjf5m~ayGDhEwKE4P7gKWPYh3uPT8vW1aGQsAi0JB~fw7qA7nlaVCpPANPIaQLo9LgiRvCEuTRaxxFgwSBXcHxFbGhZCDmUzKvSJgk6Z5TE6jDqOj2nDSDRGu2U0keA5nJu8nvyD4HwJGRDLfNyVDFMa19HDaT2Ma34wVhOSClS-7jUhr4UYwjgLVostmHaFH9OcTtgRWw~UHXSsW30Yrjao2YsDVbskurQ81cjr7SusT-RRXJUQoL1sT91utNz4oBw__',
  'https://s3-alpha-sig.figma.com/img/ff85/788d/96b4a3ec1b31b6baf36b11c772529753?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=cjUTegC0Ory6cMHPNzEqOS8K3anjn8DBCi~2j6hiEcZS1exKyuo3g0jfke26Y~HVjFa0YhUjW8tjxjd7KBmgLWHxow2U8YwP6bNJBHeNgKMwpDHk2aF1vUPO-nfDEM7ZCfZCSsWcUz~eNS5XP5XwFvd7aP8g~62hJnvdtkdMpP3qiCS70w~QUAdwBtb57TjBD3EWelFbKg2JcWTVlIxYd7qtj~uuP8-0HXrgmdssska9T3tCF0aT6REa4QFa7dYggiKL3rk8aLyjJ46RbOm9l~mqbIs9PqQakObhU~621G1PeAX9ZtkgzryhtNWG6EKyGTzhEvm4~-~AWriZFAVebw__',
  'https://s3-alpha-sig.figma.com/img/efd0/12b5/6a0078a4aa75b0e9a9fb53a6d9a7c560?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=B4Aqn-hPJ0ihHPDfRzMOJKsIyqx03layytoxWWfcKWjb5Y99kfmk79oYrVGeBG1r9Upf2S4T9qEoYombOIKwE5BQ-uHa5bHjqSrrGN-Ki4V9uwv-FoINnD1-yK8DFE75XlMlQAWIw7c19End47NjTtAF7aZwN~votmj7HtmRK5~ftLRDqNL~SuLN0-LoMGpuDPhN2DMbafK1KOZeP5ur-dZKNfLQlg7KoOk8DcfS6p1IkoLY56tYQgrDmNG9YzXPSRsjrXAO1BDvYQRw0xdepU9cc63~lYlFbQu0j15PcUKGt9BjkjLwxB5sibmX6qG4Ct4tu161kpw3~urASb84eg__',
];

interface Props {
  post: SharedPost;
}

function Item({ label, data }: { label: string; data: string }) {
  return (
    <styles.dealItemContainer>
      <p>{label}</p>
      <p>{data}</p>
    </styles.dealItemContainer>
  );
}

export function SharedPostPage({ post }: Props) {
  const [map, setMap] = useState<naver.maps.Map | null>(null);

  const onMoveToCenter = () => {
    if (map === null) return;

    const center = new naver.maps.LatLng(37.6090857, 126.9966865);
    map.setCenter(center);
  };

  useEffect(() => {
    const center = new naver.maps.LatLng(37.6090857, 126.9966865);
    setMap(new naver.maps.Map('map', { center }));
  }, []);

  return (
    <styles.container>
      <styles.houseInfo>
        <ImageGrid images={dummyImages} />
        <styles.titleRow>
          <styles.title>{post.title}</styles.title>
          <Bookmark
            hasBorder={false}
            color="#000"
            marked={false}
            onToggle={() => {}}
          />
        </styles.titleRow>
        <styles.briefInfoContainer>
          <div>
            <p>모집 1명 / 총원 2명</p>
            <p>원룸 · 방1</p>
            <p>500 / 50 / 5</p>
          </div>
          <styles.viewCount>저장 4 · 조회 22</styles.viewCount>
        </styles.briefInfoContainer>
        <HorizontalDivider />
        <styles.content>
          <styles.dealInfo>
            <styles.dealTitle>거래 정보</styles.dealTitle>
            <styles.dealContent>
              <Item label="거래방식" data="월세" />
              <Item label="보증금" data="500만원" />
              <Item label="월세" data="50만원" />
              <Item label="관리비" data="5만원" />
            </styles.dealContent>
          </styles.dealInfo>
          <styles.roomInfo>
            <styles.roomInfoTitle>방 정보</styles.roomInfoTitle>
            <styles.roomInfoContent>
              <Item label="방 종류" data="원룸" />
              <Item label="구조" data="방 1" />
              <Item label="면적" data="10평" />
              <Item label="층수" data="지상 / 2층" />
              <Item label="추가 옵션" data="주차가능, 에어컨" />
            </styles.roomInfoContent>
          </styles.roomInfo>
          <styles.detailInfoContanier>
            <styles.detailInfoTitle>상세 정보</styles.detailInfoTitle>
            <styles.detailInfoContent>
              안녕하세요! 저는 현재 룸메이트를 찾고 있는 정연수입니다. 서울시
              정릉동에서 함께 살아갈 룸메이트를 구하고 있습니다. 주로 밤에
              작업을 하며 새벽 2시~3시쯤에 취침합니다. 관심있으신 분들 연락
              주세요!
            </styles.detailInfoContent>
          </styles.detailInfoContanier>
          <styles.locationInfoContainer>
            <styles.locationInfoTitle>위치 정보</styles.locationInfoTitle>
            <styles.locationInfoContent
              onClick={() => {
                onMoveToCenter();
              }}
            >
              서울특별시 성북구 정릉로 104
            </styles.locationInfoContent>
            <styles.map id="map" />
            <styles.locationEnvInfo>
              <Item label="정릉역" data="10분" />
              <Item label="버스정류장" data="5분" />
              <Item label="국민대학교" data="20분" />
              <Item label="편의점" data="1분" />
            </styles.locationEnvInfo>
          </styles.locationInfoContainer>
        </styles.content>
      </styles.houseInfo>
      <styles.roomParticipantsContainer>
        <styles.roomParticipants>
          {dummyParticipants.map((participant, index) => (
            <MiniCircularProfileImage
              isHost={index === 0}
              key={participant}
              url={participant}
              style={{
                zIndex: dummyParticipants.length - index - 1,
                transform: `translateX(-${1.31 * index}rem)`,
              }}
            />
          ))}
        </styles.roomParticipants>
        <styles.hostInfo>
          <styles.hostInfoContainer>
            <CircularProfileImage
              diameter={110}
              percentage={50}
              url="https://s3-alpha-sig.figma.com/img/59a5/3c6f/ae49249b51c7d5d81ab89eeb0bf610f1?Expires=1711929600&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=bS1lS~7-WQsD9x5vHJBOiMnhhFjI~VgCwJH6Bzz~IxFWob-PV-XZweWFIhU6yJC3XHv5qZKZxnP9RWT~0ciIbQfofuhbODEUxnMHe6oq8Dl45khsD30dnXOK~FLBPpWhMumJO-zPpuWjiRwsZ35mfWLbgyT7dND41I9yXCyRASQx9v2iAGzDoVzTfvtkjRyGw6es6fSXRsFGMqthnzYmv7DZT~FCz2avi3-NqXruXQpkijQHNEQUM61ThFiNYEIv8vb1wZWf-USbbJpE8bdbUneblY2T0cWwMRBtKbCrJ0Y~P9lvFbzqBv7h9WOzNyJW~~KeG9vVrBmLRRo1BsNdng__"
            />
            <styles.hostInfoContent>
              <h1>김마루</h1>
              <p>24세</p>
              <p>성북 길음동</p>
            </styles.hostInfoContent>
          </styles.hostInfoContainer>
          <styles.hostCardContent>
            <p className="essential">비흡연</p>
            <p>새벽형</p>
            <p>친구초대 괜찮아요</p>
            <p className="essential">실내취식 괜찮아요</p>
          </styles.hostCardContent>
          <styles.hostButtonsContainer>
            <button type="button" className="color">
              채팅하기
            </button>
            <div>
              <button type="button">프로필 보기</button>
              <Bookmark
                color="#888"
                hasBorder
                marked={false}
                onToggle={() => {}}
              />
            </div>
          </styles.hostButtonsContainer>
          <styles.cardToggleButtonContainer>
            <CardToggleButton label="마이카드" />
            <CardToggleButton label="메이트카드" />
          </styles.cardToggleButtonContainer>
        </styles.hostInfo>
      </styles.roomParticipantsContainer>
    </styles.container>
  );
}
