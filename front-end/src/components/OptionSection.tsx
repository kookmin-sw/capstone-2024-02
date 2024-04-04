'use client';

import React, { useState } from 'react';
import styled from 'styled-components';

const styles = {
  optionContainer: styled.div`
    display: flex;
    flex-direction: column;
    width: 100%;
  `,
  optionDescription: styled.p`
    color: #9a95a3;

    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,
  optionList: styled.ul`
    width: 100%;
    margin-top: 2.25rem;
  `,
  optionListItem: styled.li`
    display: flex;
    align-items: center;
    gap: 2rem;
    margin-bottom: 1.25rem;
  `,
  optionListImg: styled.img`
    width: 3.125rem;
    height: 3.125rem;
  `,
  optionListCheckItemContainer: styled.div`
    width: 100%;
    display: flex;
    align-items: flex-start;
    align-content: flex-start;
    gap: 0.5rem;
    flex-wrap: wrap;
    position: relative;
  `,
  personalContainer: styled.div`
    width: 22rem;
    display: flex;
    align-items: flex-start;
    align-content: flex-start;
    gap: 0.5rem;
    flex-wrap: wrap;
  `,
  cleanTestContainer: styled.div`
    position: absolute;
    left: 22.19rem;
    height: 2.44rem;
    display: flex;
    align-items: center;
    cursor: pointer;

    background-image: url('/option-img/Keyboard arrow down.svg');
    background-repeat: no-repeat;
    background-position: left center;
  `,
  cleanTestDescription: styled.p`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    cursor: pointer;
    margin-left: 2rem;
  `,
  budgetContainer: styled.div`
    display: flex;
    align-items: flex-end;
    gap: 1.75rem;
    width: 100%;
  `,
  budgetBar: styled.div`
    width: 25rem;
    height: 0.3125rem;
    flex-shrink: 0;
    border-radius: 1.25rem;
    background: #d9d9d9;
    margin-bottom: 0.63rem;
  `,
  mbtiSection: styled.div`
    display: inline-flex;
    align-items: flex-end;
    gap: 2rem;
    margin: 0.5rem 0;
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
    border-radius: 1.5rem;
  `,
  sliderDot: styled.span`
    position: absolute;
    cursor: pointer;
    top: 0.25rem;
    left: 0.25rem;
    bottom: 0.25rem;
    background-color: white;
    -webkit-transition: 0.4s;
    transition: 0.4s;
    border-radius: 50%;
    width: 1rem;
    height: 1rem;
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

interface CheckItemProps {
  $isSelected: boolean;
}

const CheckItem = styled.div<CheckItemProps>`
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

const LivingPatternOptions = ['아침형', '올빼미형'];
const EatingOptions = ['실내취식 싫어요', '야식 안먹어요', '음주'];
const HearingOptions = [
  '잠버릇 있어요',
  '알람 잘 못 들어요',
  '잠귀 밝아요',
  '게임 소음 허용',
];
const WeatherOptions = ['더위 많이 타요', '추위 많이 타요'];
const CleanOptions = ['상', '평범보통', '천하태평'];
const PersonalOptions = [
  '반려동물',
  '차량 보유',
  '집에서 활동',
  '밖에서 활동',
  '친구초대 허용',
  '취미 같이 즐겨요',
  '엠비티아이',
  '전공',
];
const BudgetOptions = ['보증금', '월세'];

export function OptionSection() {
  type SelectedOptions = Record<string, boolean>;

  const [selectedOptions, setSelectedOptions] = useState<SelectedOptions>({});

  const [toggleStates, setToggleStates] = useState({
    toggle1: false,
    toggle2: false,
    toggle3: false,
    toggle4: false,
  });

  const handleOptionClick = (option: string) => {
    setSelectedOptions(prevSelectedOptions => ({
      ...prevSelectedOptions,
      [option]: !prevSelectedOptions[option],
    }));
  };

  const toggleSwitch = (toggleName: keyof typeof toggleStates) => {
    setToggleStates(prevState => ({
      ...prevState,
      [toggleName]: !prevState[toggleName],
    }));
  };

  return (
    <styles.optionContainer>
      <styles.optionDescription>선택</styles.optionDescription>
      <styles.optionList>
        <styles.optionListItem>
          <styles.optionListImg src="/option-img/visibility.svg" />
          <styles.optionListCheckItemContainer>
            {LivingPatternOptions.map(option => (
              <CheckItem
                key={option}
                $isSelected={selectedOptions[option]}
                onClick={() => {
                  handleOptionClick(option);
                }}
              >
                {option}
              </CheckItem>
            ))}
          </styles.optionListCheckItemContainer>
        </styles.optionListItem>
        <styles.optionListItem>
          <styles.optionListImg src="/option-img/restaurant.svg" />
          <styles.optionListCheckItemContainer>
            {EatingOptions.map(option => (
              <CheckItem
                key={option}
                $isSelected={selectedOptions[option]}
                onClick={() => {
                  handleOptionClick(option);
                }}
              >
                {option}
              </CheckItem>
            ))}
          </styles.optionListCheckItemContainer>
        </styles.optionListItem>
        <styles.optionListItem>
          <styles.optionListImg src="/option-img/hearing.svg" />
          <styles.optionListCheckItemContainer>
            {HearingOptions.map(option => (
              <CheckItem
                key={option}
                $isSelected={selectedOptions[option]}
                onClick={() => {
                  handleOptionClick(option);
                }}
              >
                {option}
              </CheckItem>
            ))}
          </styles.optionListCheckItemContainer>
        </styles.optionListItem>
        <styles.optionListItem>
          <styles.optionListImg src="/option-img/device_thermostat.svg" />
          <styles.optionListCheckItemContainer>
            {WeatherOptions.map(option => (
              <CheckItem
                key={option}
                $isSelected={selectedOptions[option]}
                onClick={() => {
                  handleOptionClick(option);
                }}
              >
                {option}
              </CheckItem>
            ))}
          </styles.optionListCheckItemContainer>
        </styles.optionListItem>
        <styles.optionListItem>
          <styles.optionListImg src="/option-img/mop.svg" />
          <styles.optionListCheckItemContainer>
            <styles.cleanTestContainer>
              <styles.cleanTestDescription>
                테스트 하기
              </styles.cleanTestDescription>
            </styles.cleanTestContainer>
            {CleanOptions.map(option => (
              <CheckItem
                key={option}
                $isSelected={selectedOptions[option]}
                onClick={() => {
                  handleOptionClick(option);
                }}
              >
                {option}
              </CheckItem>
            ))}
          </styles.optionListCheckItemContainer>
        </styles.optionListItem>
        <styles.optionListItem>
          <styles.optionListImg src="/option-img/person.svg" />
          <styles.personalContainer>
            {PersonalOptions.map(option => (
              <CheckItem
                key={option}
                $isSelected={selectedOptions[option]}
                onClick={() => {
                  handleOptionClick(option);
                }}
              >
                {option === '엠비티아이' ? <>MBTI</> : option}
              </CheckItem>
            ))}
            {selectedOptions['엠비티아이'] ? (
              <styles.mbtiSection>
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
            ) : (
              <></>
            )}
          </styles.personalContainer>
        </styles.optionListItem>
        <styles.optionListItem>
          <styles.optionListImg src="/option-img/home_work.svg" />
          <styles.optionListCheckItemContainer>
            {BudgetOptions.map(option => (
              <styles.budgetContainer key={option}>
                <CheckItem
                  $isSelected={selectedOptions[option]}
                  onClick={() => {
                    handleOptionClick(option);
                  }}
                >
                  {option}
                </CheckItem>
                <styles.budgetBar />
              </styles.budgetContainer>
            ))}
          </styles.optionListCheckItemContainer>
        </styles.optionListItem>
      </styles.optionList>
    </styles.optionContainer>
  );
}
