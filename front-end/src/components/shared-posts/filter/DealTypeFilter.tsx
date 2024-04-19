'use client';

import styled from 'styled-components';

import { RangeSlider } from '@/components';
import { useSharedPostsFilter } from '@/entities/shared-posts-filter';

const styles = {
  container: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;

    gap: 2rem;
    border-radius: 20px;
    background: #fff;

    width: 20rem;
  `,
  item: styled.div`
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;

    h1 {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1.125rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;
    }

    div {
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;

      button {
        all: unset;
        cursor: pointer;

        display: flex;
        padding: 0.75rem 1.25rem;
        justify-content: center;
        align-items: center;
        gap: 0.5rem;

        border-radius: 8px;
        border: 1px solid #000;

        color: #000;
        font-family: 'Noto Sans KR';
        font-size: 1rem;
        font-style: normal;
        font-weight: 400;
        line-height: normal;

        transition:
          150ms border ease-in-out,
          150ms background-color ease-in-out,
          150ms color ease-in-out;
      }

      .selected {
        background-color: #e15637;
        color: white;

        border: 1px solid white;
      }
    }
  `,
};

export function DealTypeFilter() {
  const { filter, setFilter } = useSharedPostsFilter();

  return (
    <styles.container>
      <styles.item>
        <h1>거래 방식</h1>
        <div>
          <button
            type="button"
            className={filter?.dealInfo?.dealType === '전세' ? 'selected' : ''}
            onClick={() => {
              setFilter(prev => ({
                ...prev,
                dealInfo: { ...prev.dealInfo, dealType: '전세' },
              }));
            }}
          >
            전세
          </button>
          <button
            type="button"
            className={filter?.dealInfo?.dealType === '월세' ? 'selected' : ''}
            onClick={() => {
              setFilter(prev => ({
                ...prev,
                dealInfo: { ...prev.dealInfo, dealType: '월세' },
              }));
            }}
          >
            월세
          </button>
        </div>
      </styles.item>
      <styles.item>
        <h1>희망 월 분담금</h1>
        <RangeSlider
          min={0}
          max={3500000}
          step={50000}
          onChange={({ low, high }) => {
            setFilter(prev => ({
              ...prev,
              dealInfo: { ...prev.dealInfo, expectedFee: { low, high } },
            }));
          }}
        />
      </styles.item>
    </styles.container>
  );
}
