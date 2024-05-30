'use client';

import Link from 'next/link';
import styled from 'styled-components';

const styles = {
  container: styled.div`
    display: flex;
    height: 100vh;
    width: 100vw;
    padding: 2rem 3rem;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    align-self: stretch;

    h1 {
      color: #494949;

      text-align: center;
      font-family: 'Noto Sans KR';
      font-size: 1.5rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
    }
  `,

  title: styled.p`
    color: var(--Main-1, #e15637);
    font-family: 'Baloo 2';
    font-size: 1.875rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
};

export function ErrorPage() {
  return (
    <styles.container>
      <h1>존재하지 않는 페이지입니다.</h1>
      <styles.title>
        <Link href="/">maru</Link>
      </styles.title>
    </styles.container>
  );
}
