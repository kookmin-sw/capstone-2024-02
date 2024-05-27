'use client';

import { useEffect, useRef } from 'react';
import styled from 'styled-components';

const styles = {
  container: styled.div`
    position: fixed;
    width: 50dvw;
    height: 50dvh;
    background-color: white;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    border-radius: 16px;
    padding: 2rem;

    top: 50%;
    left: 50%;
    translate: -50% -50%;

    display: flex;
    flex-direction: column;
    gap: 1rem;

    @media (max-width: 768px) {
      padding: 1rem;
      width: 60dvw;
    }
  `,
  row: styled.div`
    display: flex;
    flex-direction: row;
    gap: 0.5rem;
    align-items: end;

    @media (max-width: 768px) {
      flex-wrap: wrap;
    }
  `,
  title: styled.h1`
    font-size: 1.25rem;

    @media (max-width: 768px) {
      font-size: 1rem;
    }
  `,
  caption: styled.span`
    font-size: 0.85rem;
    color: #808080;
  `,
};

export function MateSearchBox({ setHidden }: { setHidden: () => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const clickHandler = (event: MouseEvent) => {
      if (
        containerRef.current !== null &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setHidden();
      }
    };

    const keyHandler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setHidden();
    };

    document.addEventListener('click', clickHandler);
    document.addEventListener('keydown', keyHandler);
    return () => {
      document.removeEventListener('click', clickHandler);
      document.removeEventListener('keydown', keyHandler);
    };
  }, []);

  return (
    <styles.container ref={containerRef}>
      <styles.row>
        <styles.title>메이트 찾기</styles.title>
        <styles.caption>(서로 팔로우된 경우에만 보입니다.)</styles.caption>
      </styles.row>
    </styles.container>
  );
}
