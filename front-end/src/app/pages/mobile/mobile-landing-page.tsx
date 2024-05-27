'use client';

import styled from 'styled-components';

const styles = {
  container: styled.div`
    display: flex;
    width: 100vw;
    height: 47.8125rem;
    padding-bottom: 2rem;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
  `,
  img: styled.img`
    width: 24.375rem;
    height: 27.125rem;
    flex-shrink: 0;
    object-fit: contain;
  `,
  description: styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.75rem;
    width: 100%;
    flex-shrink: 0;

    h1 {
      color: #000;

      font-family: 'Noto Sans KR';
      font-size: 2rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
    }
    p {
      color: #000;

      text-align: center;
      font-family: 'Noto Sans KR';
      font-size: 1.5rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;
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
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    align-self: stretch;
  `,
  section3: styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 1rem;
  `,
};

export function MobileLandingPage() {
  return (
    <styles.container>
      <styles.img alt="Brazuca Date Night" src="/landing-page-image.png" />
      <styles.description>
        <h1>공동주거생활의 A to Z</h1>
        <p>
          마루에서 여러분만을 위한
          <br /> 메이트를 만나보세요
        </p>
      </styles.description>
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
    </styles.container>
  );
}
