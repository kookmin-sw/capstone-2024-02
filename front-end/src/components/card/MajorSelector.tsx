'use client';

import React, { useEffect, useState } from 'react';
import styled from 'styled-components';

const styles = {
  majorSelector: styled.select`
    -webkit-appearance: none;
    -moz-appearance: none;

    appearance: none;
    display: flex;
    width: 5.875rem;
    padding: 0.575rem 0;
    align-items: center;
    background: var(--White, #fff);
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    border: none;
    border-radius: 16px;

    color: var(--Gray-3, #888);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
    text-align: center;

    position: absolute;
    bottom: 0;
    left: 12rem;
    z-index: 5;

    &:focus {
      outline: none;
    }

    @media (max-width: 768px) {
      width: 3.5rem;
      border-radius: 13px;
      font-size: 0.625rem;
      left: 9.3rem;
    }
  `,
};

const majorOptions = ['공학', '교육', '인문', '사회', '자연', '예체능', '의약'];

export function MajorSelector({
  major,
  onChange,
}: {
  major: string | undefined;
  onChange: React.Dispatch<React.SetStateAction<string>>;
}) {
  const [selectedMajor, setSelectedMajor] = useState('');
  useEffect(() => {
    if (major !== undefined) setSelectedMajor(major);
  }, [major]);
  const handleMajorChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedMajor(event.target.value);
  };
  useEffect(() => {
    onChange(selectedMajor);
  }, [selectedMajor]);
  return (
    <styles.majorSelector
      value={selectedMajor ?? ''}
      onChange={handleMajorChange}
    >
      <option value="">선택</option>
      {majorOptions.map(option => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </styles.majorSelector>
  );
}
