'use client';

import { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';

import { useIsMobile } from '@/shared/mobile';

const styles = {
  container: styled.div`
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;

    position: relative;
    width: fit-content;
    height: fit-content;

    display: flex;
    flex-direction: column;
    padding: 0.75rem 1.25rem;

    border-radius: 12px;
    border: 2px solid var(--Gray-4, #dfdfdf);

    cursor: pointer;

    @media (max-width: 768px) {
      padding: 0.25rem 1rem;
    }
  `,
  title: styled.div`
    display: flex;
    flex-direction: row;
    gap: 0.5rem;

    color: var(--Gray-3, #888);
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    @media (max-width: 768px) {
      font-size: 0.75rem;
    }
  `,
  content: styled.div<{ $hidden: boolean }>`
    visibility: ${({ $hidden }) => ($hidden ? 'hidden' : 'visible')};

    border-radius: 20px;
    background: #fff;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);

    position: absolute;
    left: 0;
    top: calc(100% + 1rem);
    z-index: 10;

    padding: 2.31rem;

    @media (max-width: 768px) {
      padding: 1.8rem;
    }
  `,
};

interface Props {
  title: string;
  children: React.ReactNode;
}

export function SharedPostFilterItem({
  title,
  children: filterComponent,
}: Props) {
  const [hidden, setHidden] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const handler = (event: MouseEvent) => {
      if (
        containerRef.current !== null &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setHidden(true);
      }
    };

    document.addEventListener('click', handler);
    return () => {
      document.removeEventListener('click', handler);
    };
  }, []);

  const handleContainerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    e.stopPropagation();
    setHidden(!hidden);
  };

  const isMobile = useIsMobile();

  return (
    <styles.container onClick={handleContainerClick} ref={containerRef}>
      <styles.title>
        {title}
        <img
          style={{
            width: isMobile ? '1.2rem' : undefined,
            height: isMobile ? '1.2rem' : undefined,
          }}
          alt="drop-down-button"
          src="/icon-drop-down_2.svg"
        />
      </styles.title>
      <styles.content
        $hidden={hidden}
        onClick={e => {
          e.stopPropagation();
        }}
      >
        {filterComponent}
      </styles.content>
    </styles.container>
  );
}
