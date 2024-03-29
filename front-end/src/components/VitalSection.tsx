'use client';

import React, { useState } from 'react';
import styled from 'styled-components';

const styles = {
  vitalContainer: styled.div`
    display: flex;
    flex-direction: column;
    width: 100%;
  `,
  vitalDescription: styled.p`
    color: var(--Main-1, #e15637);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,
  vitalListContainer: styled.div`
    display: flex;
    margin-top: 2.62rem;
    gap: 3.12rem;
  `,
  vitalList: styled.ul`
    display: inline-flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2.5rem;
    flex-shrink: 0;
  `,
  vitalListItem: styled.li`
    color: #000;

    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    list-style-type: none;
  `,
  vitalCheckList: styled.ul`
    display: inline-flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1.5rem;
  `,
  vitalCheckListItem: styled.li`
    display: flex;
    list-style-type: none;
  `,

  birthYear: styled.select`
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;

    width: 6.125rem;
    height: 3.125rem;
    display: inline-flex;
    padding: 0.75rem 1rem;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    border: 2px solid var(--Gray-4, #dfdfdf);
    background: #fff;

    color: var(--Gray-3, #888);
    font-family: 'Noto Sans KR';
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    background-image: url('/drop-icon.svg');
    background-repeat: no-repeat;
    background-position: calc(100% - 0.6875rem) center;

    padding-right: 2.75rem;
  `,

  searchBox: styled.div`
    display: flex;
    padding: 0.875rem 3.0625rem 0.8125rem 1.1875rem;
    align-items: center;
    border: 2px solid var(--Gray-4, #dfdfdf);
    background: var(--White, #fff);
  `,
  mapInput: styled.input`
    color: var(--Gray-3, #888);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
    border: none;
    &:focus {
      outline: none;
    }
  `,
};

interface CheckItemProps {
  isSelected: boolean;
}

const CheckItem = styled.div<CheckItemProps>`
  margin-right: 0.5rem;
  display: flex;
  padding: 0.5rem 1.5rem;
  justify-content: center;
  align-items: center;
  gap: 0.5rem;
  border-radius: 1.625rem;
  border: 2px solid #dfdfdf;
  background: #fff;
  cursor: pointer;

  color: #888;

  font-family: 'Noto Sans KR';
  font-size: 1rem;
  font-style: normal;
  font-weight: 500;
  line-height: normal;

  ${props =>
    props.isSelected
      ? {
          color: 'var(--Main-1, #E15637)',
          border: '2px solid var(--Main-1, #E15637)',
        }
      : {
          backgroundColor: '#FFF',
          color: '#888',
        }};
`;

const years = Array.from({ length: 100 }, (_, index) => 2024 - index);

export function VitalSection() {
  interface SelectedState {
    gender: string | null;
    smoking: string | null;
    room: string | null;
  }

  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const [selectedState, setSelectedState] = useState<SelectedState>({
    gender: null,
    smoking: null,
    room: null,
  });

  function handleOptionClick(
    optionName: keyof SelectedState,
    item: string | number,
  ) {
    setSelectedState(prevState => ({
      ...prevState,
      [optionName]: prevState[optionName] === item ? null : item,
    }));
  }

  const handleYearChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const yearValue = parseInt(event.target.value, 10);
    setSelectedYear(Number.isNaN(yearValue) ? null : yearValue);
  };
  return (
    <styles.vitalContainer>
      <styles.vitalDescription>필수</styles.vitalDescription>
      <styles.vitalListContainer>
        <styles.vitalList>
          <styles.vitalListItem>성별</styles.vitalListItem>
          <styles.vitalListItem>출생 연도</styles.vitalListItem>
          <styles.vitalListItem>흡연 여부</styles.vitalListItem>
          <styles.vitalListItem>메이트와</styles.vitalListItem>
          <styles.vitalListItem>희망 지역</styles.vitalListItem>
        </styles.vitalList>
        <styles.vitalCheckList>
          <styles.vitalCheckListItem>
            <CheckItem
              isSelected={selectedState.gender === '남성'}
              onClick={() => {
                handleOptionClick('gender', '남성');
              }}
            >
              남성
            </CheckItem>
            <CheckItem
              isSelected={selectedState.gender === '여성'}
              onClick={() => {
                handleOptionClick('gender', '여성');
              }}
            >
              여성
            </CheckItem>
          </styles.vitalCheckListItem>
          <styles.vitalCheckListItem>
            <styles.birthYear
              value={selectedYear ?? ''}
              onChange={handleYearChange}
            >
              <option value="">년도</option>
              {years.map(year => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </styles.birthYear>
          </styles.vitalCheckListItem>
          <styles.vitalCheckListItem>
            <CheckItem
              isSelected={selectedState.smoking === '흡연'}
              onClick={() => {
                handleOptionClick('smoking', '흡연');
              }}
            >
              흡연
            </CheckItem>
            <CheckItem
              isSelected={selectedState.smoking === '비흡연'}
              onClick={() => {
                handleOptionClick('smoking', '비흡연');
              }}
            >
              비흡연
            </CheckItem>
          </styles.vitalCheckListItem>
          <styles.vitalCheckListItem>
            <CheckItem
              isSelected={selectedState.room === '같은 방'}
              onClick={() => {
                handleOptionClick('room', '같은 방');
              }}
            >
              같은 방
            </CheckItem>
            <CheckItem
              isSelected={selectedState.room === '다른 방'}
              onClick={() => {
                handleOptionClick('room', '다른 방');
              }}
            >
              다른 방
            </CheckItem>
            <CheckItem
              isSelected={selectedState.room === '상관없어요'}
              onClick={() => {
                handleOptionClick('room', '상관없어요');
              }}
            >
              상관없어요
            </CheckItem>
          </styles.vitalCheckListItem>
          <styles.vitalCheckListItem>
            <styles.searchBox>
              <styles.mapInput placeholder="ex) 한국동,한국역,한국대학교" />
            </styles.searchBox>
          </styles.vitalCheckListItem>
        </styles.vitalCheckList>
      </styles.vitalListContainer>
    </styles.vitalContainer>
  );
}
