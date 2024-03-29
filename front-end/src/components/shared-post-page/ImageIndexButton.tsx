'use client';

import styled from 'styled-components';

const styles = {
  container: styled.div<{ $disabled: boolean }>`
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;

    width: 3.125rem;
    height: 3.12481rem;
    flex-shrink: 0;

    cursor: pointer;

    opacity: ${({ $disabled }) => ($disabled ? 0.25 : 1)};

    img {
      position: absolute;
      margin: auto;
    }
  `,
};

export function ImageIndexButton({
  direction,
  disabled,
  onClick,
}: {
  direction: 'left' | 'right';
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <styles.container onClick={onClick} $disabled={disabled}>
      <img alt="" src="/icon-ellipse.svg" />
      <img alt="" src={`/icon-${direction}.svg`} />
    </styles.container>
  );
}
