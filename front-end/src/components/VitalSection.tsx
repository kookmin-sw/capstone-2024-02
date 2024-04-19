'use client';

import React, { useEffect, useState } from 'react';
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
  vitalListContainer: styled.ul`
    display: flex;
    flex-direction: column;
    margin-top: 2.62rem;
    gap: 1.5rem;
  `,
  vitalList: styled.li`
    display: inline-flex;
    align-items: center;
    gap: 2.5rem;
    flex-shrink: 0;
  `,
  vitalListItemDescription: styled.p`
    width: 5.125rem;
    color: #000;

    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    list-style-type: none;
  `,
  vitalCheckListContainer: styled.div`
    display: inline-flex;
    align-items: flex-start;
    gap: 0.5rem;
  `,

  birthYear: styled.select`
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;

    width: 6.7rem;
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
  $isSelected: boolean;
}

const CheckItem = styled.div<CheckItemProps>`
  margin-right: 0.5rem;
  display: flex;
  padding: 0.5rem 1.5rem;
  justify-content: center;
  align-items: center;
  gap: 0.5rem;
  border-radius: 26px;
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
    props.$isSelected
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

interface SelectedState {
  smoking: string | undefined;
  room: string | undefined;
}

export function VitalSection({
  gender,
  birthYear,
  smoking,
  room,
  onFeatureChange,
  isMySelf,
}: {
  gender: string | undefined;
  birthYear: string | undefined;
  smoking: string | undefined;
  room: string | undefined;
  onFeatureChange: (
    optionName: keyof SelectedState,
    item: string | number,
  ) => void;
  isMySelf: boolean;
}) {
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const [selectedState, setSelectedState] = useState<SelectedState>({
    smoking: undefined,
    room: undefined,
  });
  useEffect(() => {
    setSelectedState({
      ...selectedState,
      smoking: smoking,
      room: room,
    });
  }, [smoking, room]);

  function handleOptionClick(
    optionName: keyof SelectedState,
    item: string | number,
  ) {
    setSelectedState(prevState => ({
      ...prevState,
      [optionName]: prevState[optionName] === item ? null : item,
    }));
    onFeatureChange(optionName, item);
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
          <styles.vitalListItemDescription>
            성별
          </styles.vitalListItemDescription>
          <styles.vitalCheckListContainer>
            <CheckItem
              $isSelected={gender === 'MALE'}
              style={{
                border: gender === 'MALE' ? 'none' : '',
                background: gender === 'MALE' ? 'var(--Gray-5, #828282)' : '',
                color: gender === 'MALE' ? '#fff' : '',
              }}
            >
              남성
            </CheckItem>
            <CheckItem
              $isSelected={gender === 'FEMALE'}
              style={{
                border: gender === 'FEMALE' ? 'none' : '',
                background: gender === 'FEMALE' ? 'var(--Gray-5, #828282)' : '',
                color: gender === 'FEMALE' ? '#fff' : '',
              }}
            >
              여성
            </CheckItem>
          </styles.vitalCheckListContainer>
        </styles.vitalList>
        <styles.vitalList>
          <styles.vitalListItemDescription>
            희망 지역
          </styles.vitalListItemDescription>
          <styles.searchBox>
            <styles.mapInput
              placeholder="ex) 한국동,한국역,한국대학교"
              readOnly={!isMySelf}
            />
          </styles.searchBox>
        </styles.vitalList>
        <styles.vitalList>
          <styles.vitalListItemDescription>
            흡연 여부
          </styles.vitalListItemDescription>
          <styles.vitalCheckListContainer>
            <CheckItem
              $isSelected={selectedState.smoking === '흡연'}
              onClick={() => {
                if (isMySelf) {
                  handleOptionClick('smoking', '흡연');
                }
              }}
            >
              흡연
            </CheckItem>
            <CheckItem
              $isSelected={selectedState.smoking === '비흡연'}
              onClick={() => {
                if (isMySelf) {
                  handleOptionClick('smoking', '비흡연');
                }
              }}
            >
              비흡연
            </CheckItem>
            <CheckItem
              $isSelected={selectedState.smoking === '상관없어요'}
              onClick={() => {
                if (isMySelf) {
                  handleOptionClick('smoking', '상관없어요');
                }
              }}
            >
              상관없어요
            </CheckItem>
          </styles.vitalCheckListContainer>
        </styles.vitalList>
        <styles.vitalList>
          <styles.vitalListItemDescription>
            메이트와
          </styles.vitalListItemDescription>
          <styles.vitalCheckListContainer>
            <CheckItem
              $isSelected={selectedState.room === '같은 방'}
              onClick={() => {
                if (isMySelf) {
                  handleOptionClick('room', '같은 방');
                }
              }}
            >
              같은 방
            </CheckItem>
            <CheckItem
              $isSelected={selectedState.room === '다른 방'}
              onClick={() => {
                if (isMySelf) {
                  handleOptionClick('room', '다른 방');
                }
              }}
            >
              다른 방
            </CheckItem>
            <CheckItem
              $isSelected={selectedState.room === '상관없어요'}
              onClick={() => {
                if (isMySelf) {
                  handleOptionClick('room', '상관없어요');
                }
              }}
            >
              상관없어요
            </CheckItem>
          </styles.vitalCheckListContainer>
        </styles.vitalList>
        <styles.vitalList>
          <styles.vitalListItemDescription>
            출생 연도
          </styles.vitalListItemDescription>
          <styles.birthYear
            value={selectedYear ?? birthYear}
            onChange={handleYearChange}
            disabled={birthYear !== undefined || !isMySelf}
          >
            <option value="">년도</option>
            {years.map(year => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </styles.birthYear>
        </styles.vitalList>
      </styles.vitalListContainer>
    </styles.vitalContainer>
  );
}
