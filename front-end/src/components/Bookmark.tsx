'use client';

import styled from 'styled-components';

const styles = {
  container: styled.div<{ $hasBorder: boolean }>`
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;

    cursor: pointer;

    ${({ $hasBorder }) =>
      $hasBorder
        ? `display: inline-flex;
        padding: 0.6875rem 1.5rem;
        border-radius: 8px;
        border: 1px solid var(--Gray-3, #888);`
        : ``}
  `,
};

interface Props {
  marked: boolean;
  onToggle: () => void;
}

function Icon({ fill }: { fill: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
    >
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M17 3H7C5.9 3 5 3.9 5 5V21L12 18L19 21V5C19 3.9 18.1 3 17 3ZM17 18L12 15.82L7 18V5H17V18Z"
        fill={fill}
      />
    </svg>
  );
}

export function Bookmark({
  marked,
  onToggle,
  color,
  hasBorder,
}: Props & { hasBorder: boolean; color: string }) {
  return (
    <styles.container
      $hasBorder={hasBorder}
      onClick={() => {
        onToggle();
      }}
    >
      <Icon fill={marked ? 'red' : color} />
    </styles.container>
  );
}
