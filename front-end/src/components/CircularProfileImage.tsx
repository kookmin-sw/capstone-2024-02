'use client';

import styled from 'styled-components';

import { CircularProgressBar } from './CircularProgressBar';

const styles = {
  backgroundContainer: styled.div<{ $url: string }>`
    width: 100%;
    height: 100%;
    border-radius: 50%;
    overflow: hidden;
    background-repeat: no-repeat;
    background-size: 80%;
    background-image: ${({ $url }) => `url("${$url}")`};
    background-position: center;
    position: absolute;
    top: 0;
    left: 0;
    z-index: 0;
  `,
  container: styled.div<{ $diameter: number }>`
    width: ${({ $diameter }) => $diameter}px;
    height: ${({ $diameter }) => $diameter}px;
    position: relative;
  `,
  CircularProgressBar: styled(CircularProgressBar)`
    position: relative;
    z-index: 1;
  `,
  percentage: styled.div`
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;

    width: 3rem;
    height: 3rem;
    flex-shrink: 0;

    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;

    border-radius: 200px;
    background: rgba(255, 255, 255, 0.8);
    box-shadow: 0px 4px 4px 0px rgba(0, 0, 0, 0.04);
    color: #e15637;

    font-family: Pretendard;
    font-size: 1rem;
    font-style: normal;
    font-weight: 600;
    line-height: 1.5rem;

    top: -40%;
    left: 65%;

    z-index: 2;

    @media (max-width: 768px) {
      width: 1.6875rem;
      height: 1.75rem;
      padding: 0.75rem 0.375rem;
      font-size: 0.75rem;
    }
  `,
};

export function CircularProfileImage({
  url,
  percentage,
  diameter,
}: {
  url: string;
  percentage: number;
  diameter: number;
}) {
  return (
    <styles.container $diameter={diameter}>
      <styles.backgroundContainer $url={url} />
    </styles.container>
  );
}
