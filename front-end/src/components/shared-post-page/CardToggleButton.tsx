'use client';

import styled from 'styled-components';

const styles = {
  container: styled.div`
    display: flex;
    gap: 0.75rem;
    width: 100%;

    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    cursor: pointer;

    align-items: center;

    div {
      display: flex;
      align-items: center;
    }

    user-select: none;
    -ms-user-select: none;
    -moz-user-select: none;
    -webkit-user-select: none;
  `,
};

function ToggleIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
    >
      <path d="M19 9L5 9L12 16L19 9Z" fill="black" />
    </svg>
  );
}

export function CardToggleButton({
  label,
  isOpen,
  onClick,
}: {
  label: string;
  isOpen: boolean;
  onClick: () => void;
}) {
  return (
    <styles.container onClick={onClick}>
      <div
        style={{
          transition: '0.2s transform ease-in-out',
          transform: isOpen ? 'rotate(0)' : 'rotate(-90deg)',
        }}
      >
        <ToggleIcon />
      </div>
      {label}
    </styles.container>
  );
}
