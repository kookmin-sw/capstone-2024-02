'use client';

import React, { useEffect, useState } from 'react';
import styled from 'styled-components';

const styles = {
  mbtiSection: styled.div`
    position: absolute;
    bottom: -7rem;
    padding: 2rem;
    border-radius: 16px;
    display: inline-flex;
    align-items: center;
    gap: 2rem;
    margin: 0.5rem 0;
    background: #fff;
    z-index: 5;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);

    @media (max-width: 768px) {
      gap: 1rem;
      padding: 1rem;
      width: 12.5rem;
      flex-wrap: wrap;
      bottom: -8rem;
    }
  `,
  mbtiToggleContainer: styled.div`
    display: inline-flex;
    align-items: flex-end;
    gap: 1rem;

    color: #000;

    font-family: 'Noto Sans KR';
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    @media (max-width: 768px) {
      font-size: 0.625rem;
    }
  `,

  switchContainer: styled.div`
    display: inline-flex;
    justify-content: center;
    align-items: flex-end;
    gap: 0.375rem;
  `,
  switchWrapper: styled.label`
    position: relative;
    display: inline-block;
    width: 2.5rem;
    height: 1.5rem;

    @media (max-width: 768px) {
      width: 2rem;
      height: 1rem;
    }
  `,
  switchInput: styled.input`
    opacity: 0;
    width: 0;
    height: 0;
  `,
  slider: styled.span`
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #bebebe;
    -webkit-transition: 0.4s;
    transition: 0.4s;
    border-radius: 24px;
  `,
  sliderDot: styled.span`
    position: absolute;
    cursor: pointer;
    top: 0.25rem;
    left: 0.25rem;
    background-color: white;
    -webkit-transition: 0.4s;
    transition: 0.4s;
    border-radius: 50%;
    width: 1rem;
    height: 1rem;

    @media (max-width: 768px) {
      width: 0.8rem;
      height: 0.8rem;
      top: 0.13rem;
      left: 0.13rem;
    }
  `,
  value: styled.p`
    color: #000;

    font-family: 'Noto Sans KR';
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    align-self: stretch;

    @media (max-width: 768px) {
      font-size: 0.625rem;
      width: 100%;
    }
  `,
};

interface ToggleSwitchProps {
  isChecked: boolean;
  onToggle: () => void;
}

function ToggleSwitch({ isChecked, onToggle }: ToggleSwitchProps) {
  return (
    <styles.switchContainer>
      <styles.switchWrapper>
        <styles.switchInput
          type="checkbox"
          checked={isChecked}
          onChange={onToggle}
        />
        <styles.slider
          style={{
            backgroundColor: isChecked ? '#E15637' : '#BEBEBE',
          }}
        >
          <styles.sliderDot
            style={{
              transform: isChecked ? 'translateX(1rem)' : 'translateX(0)',
            }}
          />
        </styles.slider>
      </styles.switchWrapper>
    </styles.switchContainer>
  );
}

export function MbtiToggle({
  mbti,
  onChange,
}: {
  mbti: string | undefined;
  onChange: React.Dispatch<React.SetStateAction<string>>;
}) {
  const [toggleStates, setToggleStates] = useState({
    toggle1: false,
    toggle2: false,
    toggle3: false,
    toggle4: false,
  });

  const [result, setResult] = useState('');
  useEffect(() => {
    if (mbti !== undefined) setResult(mbti);
  }, [mbti]);

  useEffect(() => {
    if (mbti !== undefined && mbti.charAt(0) === 'I') {
      setToggleStates(prevState => ({
        ...prevState,
        toggle1: true,
      }));
    } else {
      setToggleStates(prevState => ({
        ...prevState,
        toggle1: false,
      }));
    }
    if (mbti !== undefined && mbti.charAt(1) === 'S') {
      setToggleStates(prevState => ({
        ...prevState,
        toggle2: true,
      }));
    } else {
      setToggleStates(prevState => ({
        ...prevState,
        toggle2: false,
      }));
    }
    if (mbti !== undefined && mbti.charAt(2) === 'T') {
      setToggleStates(prevState => ({
        ...prevState,
        toggle3: true,
      }));
    } else {
      setToggleStates(prevState => ({
        ...prevState,
        toggle3: false,
      }));
    }
    if (mbti !== undefined && mbti.charAt(3) === 'J') {
      setToggleStates(prevState => ({
        ...prevState,
        toggle4: true,
      }));
    } else {
      setToggleStates(prevState => ({
        ...prevState,
        toggle4: false,
      }));
    }
  }, [mbti]);

  const toggleSwitch = (toggleName: keyof typeof toggleStates) => {
    setToggleStates(prevState => ({
      ...prevState,
      [toggleName]: !prevState[toggleName],
    }));
  };
  useEffect(() => {
    const mbtiString: string[] = ['E', 'N', 'F', 'P'];
    if (toggleStates.toggle1) mbtiString[0] = 'I';
    if (toggleStates.toggle2) mbtiString[1] = 'S';
    if (toggleStates.toggle3) mbtiString[2] = 'T';
    if (toggleStates.toggle4) mbtiString[3] = 'J';
    const newResult = mbtiString.join('');
    setResult(newResult);
    onChange(newResult);
  }, [toggleStates, onChange]);

  return (
    <styles.mbtiSection>
      <styles.value>{result}</styles.value>
      <styles.mbtiToggleContainer>
        E
        <ToggleSwitch
          isChecked={toggleStates.toggle1}
          onToggle={() => {
            toggleSwitch('toggle1');
          }}
        />
        I
      </styles.mbtiToggleContainer>
      <styles.mbtiToggleContainer>
        N
        <ToggleSwitch
          isChecked={toggleStates.toggle2}
          onToggle={() => {
            toggleSwitch('toggle2');
          }}
        />
        S
      </styles.mbtiToggleContainer>
      <styles.mbtiToggleContainer>
        F
        <ToggleSwitch
          isChecked={toggleStates.toggle3}
          onToggle={() => {
            toggleSwitch('toggle3');
          }}
        />
        T
      </styles.mbtiToggleContainer>
      <styles.mbtiToggleContainer>
        P
        <ToggleSwitch
          isChecked={toggleStates.toggle4}
          onToggle={() => {
            toggleSwitch('toggle4');
          }}
        />
        J
      </styles.mbtiToggleContainer>
    </styles.mbtiSection>
  );
}
