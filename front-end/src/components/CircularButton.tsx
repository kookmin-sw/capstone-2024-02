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

    cursor: ${({ $disabled }) => ($disabled ? 'auto' : 'pointer')};

    opacity: ${({ $disabled }) => ($disabled ? 0.25 : 1)};

    img {
      position: absolute;
      margin: auto;
    }

    @media (max-width: 768px) {
      width: 2.25rem;
      height: 2.25rem;
      flex-shrink: 0;
    }
  `,
};

interface Props {
  direction: 'left' | 'right';
  disabled: boolean;
  onClick: () => void;
  className?: string;
}

export function CircularButton({
  direction,
  disabled,
  onClick,
  className,
}: Props) {
  return (
    <styles.container
      className={className}
      onClick={() => {
        if (disabled) return;
        onClick();
      }}
      $disabled={disabled}
    >
      <img alt="" src="/icon-ellipse.svg" />
      <img alt="" src={`/icon-${direction}.svg`} />
    </styles.container>
  );
}
