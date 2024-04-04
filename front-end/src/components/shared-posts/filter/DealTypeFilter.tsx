'use client';

import styled from 'styled-components';

import { RangeSlider } from '@/components';

const styles = {
  container: styled.div`
    display: flex;
    flex-direction: column;
    gap: 1.56rem;

    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;

    h3 {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1.125rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;
    }
  `,
};

export function DealTypeFilter() {
  return (
    <styles.container>
      <h3>거래방식</h3>
      <h3>보증금(전세금)</h3>
      <RangeSlider min={0} max={100} onChange={() => {}} />
      <h3>월세</h3>
    </styles.container>
  );
}
