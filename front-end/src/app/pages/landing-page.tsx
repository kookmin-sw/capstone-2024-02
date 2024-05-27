'use client';

import styled from 'styled-components';
import { v4 as uuidv4 } from 'uuid';

const styles = {
  container: styled.div`
    width: 100%;
  `,
  section1: styled.div`
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    padding: 0rem 13.75rem;
  `,
  img: styled.img`
    width: 29.1875rem;
    height: 39.9375rem;
    flex-shrink: 0;
    object-fit: contain;

    margin-bottom: -6rem;
  `,
  description: styled.div`
    display: flex;
    flex-direction: column;
    align-items: end;
    gap: 1.75rem;
    width: 27.8125rem;
    margin-top: 6.25rem;
    flex-shrink: 0;

    h1 {
      color: #000;
      text-align: right;
      font-family: 'Noto Sans KR';
      font-size: 2rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
      margin-bottom: 2rem;
    }
    p {
      color: #000;
      text-align: right;
      font-family: 'Noto Sans KR';
      font-size: 1.5rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;
      margin-bottom: 6.5rem;
    }
  `,
  loginButtons: styled.div`
    display: flex;
    flex-direction: column;
    gap: 0.6875rem;

    button {
      all: unset;
      cursor: pointer;
      display: flex;
      justify-content: center;
    }

    img {
      width: 16.5625rem;
      height: 2.6875rem;
      flex-shrink: 0;
      border-radius: 8px;
      object-fit: cover;
    }
  `,
  section2: styled.div`
    position: relative;
    width: 100dvw;
    height: 31.25rem;
    flex-shrink: 0;
    background: #f7f6f9;
  `,
  section3: styled.div`
    display: flex;
    justify-content: end;

    height: 31.25rem;
    padding-right: 21.25rem;
  `,
  imageBox: styled.div`
    display: flex;
    flex-direction: row;
    gap: 1rem;
    width: 17rem;
    height: 41rem;
    flex-shrink: 0;
  `,
  boxColumn: styled.div<{ $margin: number }>`
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: ${({ $margin }) => `${$margin / 16}rem`};

    position: relative;
    top: -5rem;
  `,
  box: styled.div`
    display: flex;
    justify-content: center;
    align-items: center;

    width: 5rem;
    height: 5rem;
    flex-shrink: 0;

    border-radius: 16px;
    background: rgba(247, 246, 249, 0.5);
  `,
  section4: styled.div`
    display: flex;
    position: relative;
    width: 100dvw;
    height: 17.9375rem;
    padding: 10.0625rem 38.71875rem 5.0625rem 39.90625rem;
    justify-content: center;
    align-items: center;
    flex-shrink: 0;
    background: #f7f6f9;
  `,
  findMateButton: styled.button`
    all: unset;
    flex-shrink: 0;
    padding: 0.5rem 1.5rem;
    flex-shrink: 0;

    border-radius: 8px;
    background: #e15637;

    color: #fff;
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
    cursor: pointer;
  `,
};

const images = [
  [
    '',
    '',
    '/icon-popular.svg',
    '/icon-hanok.svg',
    '/icon-building.svg',
    '',
    '',
  ],
  [
    '',
    '',
    '/icon-building.svg',
    '/icon-countryside.svg',
    '/icon-camping.svg',
    '',
    '',
  ],
  ['', '', '/icon-house.svg', '/icon-camping.svg', 'icon-hanok.svg', '', ''],
];

export function LandingPage() {
  return (
    <styles.container>
      <styles.section1>
        <styles.img alt="Brazuca Date Night" src="/landing-page-image.png" />
        <styles.description>
          <h1>공동주거생활의 A to Z</h1>
          <p>
            마루에서 여러분만을 위한
            <br /> 메이트를 만나보세요
          </p>
          <styles.loginButtons>
            <a
              href={`${process.env.NEXT_PUBLIC_API_URL}/oauth2/authorize/kakao?redirect_uri=${process.env.NEXT_PUBLIC_CLIENT_URL}/login`}
            >
              <img alt="kakao" src="/kakao-login.png" />
            </a>
            <a
              href={`${process.env.NEXT_PUBLIC_API_URL}/oauth2/authorize/naver?redirect_uri=${process.env.NEXT_PUBLIC_CLIENT_URL}/login`}
            >
              <img alt="naver" src="/naver-login.png" />
            </a>
          </styles.loginButtons>
        </styles.description>
      </styles.section1>
      <styles.section2 />
      <styles.section3>
        <styles.imageBox>
          {images.map((column, i) => (
            <styles.boxColumn key={JSON.stringify(column)} $margin={16 * i}>
              {column.map(url =>
                url !== '' ? (
                  <styles.box key={url}>
                    <img alt={url} src={url} />
                  </styles.box>
                ) : (
                  <styles.box key={uuidv4()} />
                ),
              )}
            </styles.boxColumn>
          ))}
        </styles.imageBox>
      </styles.section3>
      <styles.section4>
        <styles.findMateButton>메이트 찾아보기</styles.findMateButton>
      </styles.section4>
    </styles.container>
  );
}
