'use client';

import styled from 'styled-components';

const styles = {
  switchContainer: styled.div`
    display: flex;
  `,
  switchWrapper: styled.label`
    position: relative;
    display: inline-block;
    width: 2.5rem;
    height: 1.5rem;
  `,
  switchInput: styled.input`
    opacity: 0;
    width: 0;
    height: 0;
  `,
  slider: styled.span<{ $isChecked: boolean }>`
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: ${({ $isChecked }) =>
      $isChecked ? '#E15637' : '#BEBEBE'};
    -webkit-transition: 0.4s transform;
    transition: 0.4s transform;
    border-radius: 24px;
  `,
  sliderDot: styled.span<{ $isChecked: boolean }>`
    position: absolute;
    cursor: pointer;
    top: 0.25rem;
    left: 0.25rem;
    bottom: 0.25rem;
    background-color: white;
    -webkit-transition: 0.4s transform;
    transition: 0.4s transform;
    border-radius: 50%;
    width: 1rem;
    height: 1rem;
    transform: ${({ $isChecked }) =>
      $isChecked ? 'translateX(1rem)' : 'translateX(0)'};
  `,
};

interface Props {
  isChecked: boolean;
  onToggle: () => void;
}

export function ToggleSwitch({ isChecked, onToggle }: Props) {
  return (
    <styles.switchContainer>
      <styles.switchWrapper>
        <styles.switchInput
          type="checkbox"
          checked={isChecked}
          onChange={onToggle}
        />
        <styles.slider $isChecked={isChecked}>
          <styles.sliderDot $isChecked={isChecked} />
        </styles.slider>
      </styles.switchWrapper>
    </styles.switchContainer>
  );
}
