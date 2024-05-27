'use client';

import styled from 'styled-components';

import {
  type AdditionalInfoType,
  useSharedPostsFilter,
} from '@/entities/shared-posts-filter';

const styles = {
  container: styled.div`
    display: flex;
    width: 31.25rem;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;

    @media (max-width: 768px) {
      width: 12rem;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    h1 {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1.125rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;

      @media (max-width: 768px) {
        font-size: 1rem;
      }
    }

    div {
      display: flex;
      align-items: flex-start;
      align-content: flex-start;
      gap: 0.5rem;
      align-self: stretch;
      flex-wrap: wrap;

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

        @media (max-width: 768px) {
          font-size: 0.75rem;
          padding: 0.5rem 1rem;
        }

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

export function ExtraInfoFilter() {
  const { filter, setFilter } = useSharedPostsFilter();

  const handleOptionClick = (option: AdditionalInfoType) => {
    setFilter(prev => {
      const value = prev.extraInfo[option] ?? false;
      return {
        ...prev,
        extraInfo: {
          ...prev.extraInfo,
          [option]: !value,
        },
      };
    });
  };

  const isSelectedChecker = (option: AdditionalInfoType) =>
    filter.extraInfo[option] === true;

  return (
    <styles.container>
      <h1>추가 옵션</h1>
      <div>
        <button
          type="button"
          className={isSelectedChecker('주차가능') ? 'selected' : ''}
          onClick={() => {
            handleOptionClick('주차가능');
          }}
        >
          주차 가능
        </button>
        <button
          type="button"
          className={isSelectedChecker('에어컨') ? 'selected' : ''}
          onClick={() => {
            handleOptionClick('에어컨');
          }}
        >
          에어컨
        </button>
        <button
          type="button"
          className={isSelectedChecker('냉장고') ? 'selected' : ''}
          onClick={() => {
            handleOptionClick('냉장고');
          }}
        >
          냉장고
        </button>
        <button
          type="button"
          className={isSelectedChecker('세탁기') ? 'selected' : ''}
          onClick={() => {
            handleOptionClick('세탁기');
          }}
        >
          세탁기
        </button>
        <button
          type="button"
          className={isSelectedChecker('베란다/테라스') ? 'selected' : ''}
          onClick={() => {
            handleOptionClick('베란다/테라스');
          }}
        >
          베란다/테라스
        </button>
      </div>
    </styles.container>
  );
}
