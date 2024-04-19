'use client';

import Link from 'next/link';
import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

import { useAuthValue, useUserData } from '@/features/auth';
import { useProfileData } from '@/features/profile';
import { getAge } from '@/shared';

const styles = {
  pageContainer: styled.div`
    display: flex;
    flex-direction: column;
    padding: 0 1.5rem;
  `,

  userProfileContainer: styled.div`
    display: inline-flex;
    align-items: flex-start;
    flex-shrink: 0;
    margin-top: 5.12rem;
  `,
  userProfileWithoutSwitch: styled.div`
    display: inline-flex;
    align-items: center;
    gap: 2.625rem;
  `,
  userPicContainer: styled.div`
    display: flex;
    width: 8.3125rem;
    height: 8.3125rem;
    justify-content: center;
    align-items: center;
    border-radius: 100px;
    border: 1px solid #dcddea;

    background: #c4c4c4;
  `,
  userPic: styled.img`
    width: 100%;
    height: 100%;
    border-radius: inherit;
    object-fit: cover;
    border: 0;
  `,
  userInfoContainer: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  `,
  userDetailedContainer: styled.div`
    display: inline-flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.25rem;
  `,
  userName: styled.div`
    color: #000;

    font-family: 'Noto Sans KR';
    font-size: 1.5rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,
  userDetailedInfo: styled.p`
    color: #000;

    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,

  switchContainer: styled.div`
    display: inline-flex;
    justify-content: center;
    align-items: flex-end;
    margin: 1.25rem 0 0 2.63rem;
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
    border-radius: 24px;
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
  switchDescription: styled.p`
    color: var(--Gray-3, #888);
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,

  authContainer: styled.div`
    height: 2rem;
    width: 5.3125rem;
    border-radius: 26px;
    background: #5c6eb4;
    margin: 1rem 1.4375rem 0 1.5625rem;
    cursor: pointer;

    display: inline-flex;
    padding: 0.25rem 0.5rem;
    justify-content: center;
    align-items: center;
    gap: 0.25rem;
  `,
  authDescription: styled.p`
    color: #fff;

    font-family: Pretendard;
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 600;
    line-height: 1.5rem;
  `,
  authCheckImg: styled.img`
    width: 1rem;
    height: 1rem;
  `,

  cardSection: styled.div`
    display: inline-flex;
    gap: 11.5rem;
    margin: 4.75rem 0 0 0;
  `,
  cardWrapper: styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    gap: 2.25rem;
    flex-shrink: 0;
  `,
  cardDescriptionSection: styled.div`
    margin: 0 1.125rem 0 4.9375rem;
    width: 36.9375rem;
    height: 2.875rem;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 20rem;
  `,
  description32px: styled.p`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 2rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
  description24px: styled.p`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1.5rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
  addButton: styled.button`
    display: flex;
    width: 6.1875rem;
    padding: 0.5rem 1.5rem;
    justify-content: center;
    align-items: center;
    gap: 0.25rem;
    border-radius: 8px;
    border: 1px solid var(--Gray-5, #828282);
    background: var(--White, #fff);
    cursor: pointer;

    color: var(--Gray-5, #828282);
    font-family: Pretendard;
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 600;
    line-height: 1.5rem;
  `,
  mateCardsContainer: styled.div`
    display: flex;
    gap: 1.69rem;
    align-items: center;
  `,
  mateCards: styled.div`
    display: flex;
    width: 35.6rem;
    overflow-x: hidden;
    scroll-behavior: smooth;
    padding: 1.5rem;
    gap: 2.88rem;
  `,
  cardContainer: styled.div`
    padding: 0 0 0 1.5rem;
    display: inline-flex;
    align-items: center;
    width: 15rem;
    height: 15rem;
    flex: 0 0 auto;
    border-radius: 20px;
    border: 1pxs olid var(--background, #f7f6f9);
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    background: var(--grey-100, #fff);
    position: relative;
  `,
  cardName: styled.p`
    color: var(--grey-900, #494949);
    font-family: 'Noto Sans KR';
    font-size: 1.375rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
  cardDefault: styled.div`
    color: #fff;
    text-align: center;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 700;
    line-height: 2.5625rem;

    width: 6.0625rem;
    height: 2.5625rem;
    flex-shrink: 0;
    border-radius: 20px 0 0 20px;
    background: var(--Main-1, #e15637);

    position: absolute;
    right: 0;
    bottom: 1.5rem;
  `,
  nextButton: styled.button`
    width: 3.125rem;
    height: 3.12481rem;
    flex-shrink: 0;
    background-image: url('/next-button.svg');
    background-repeat: no-repeat;
    border: none;
    background-color: #fff;
    cursor: pointer;
  `,
  prevButton: styled.button`
    width: 3.125rem;
    height: 3.12481rem;
    flex-shrink: 0;
    background-image: url('/prev-button.svg');
    background-repeat: no-repeat;
    border: none;
    background-color: #fff;
    cursor: pointer;
  `,

  maruContainer: styled.div`
    margin-top: 9.5625rem;
    display: flex;
    flex-direction: column;
  `,

  weekContainer: styled.div`
    display: flex;
    width: 70.25rem;
    height: 15.625rem;
    justify-content: center;
    align-items: flex-start;
    gap: 1.5rem;
    flex-shrink: 0;
    margin: 3.1875rem 0 4.0625rem 1.6875rem;
  `,
  dayContainer: styled.div`
    width: 8.75rem;
    height: 15.625rem;
    padding: 1.5rem 0;
    flex-shrink: 0;
    border-radius: 20px;
    background: var(--background, #f7f6f9);

    justify-content: center;
    flex-direction: column;
  `,
  day: styled.p`
    color: #000;

    text-align: center;
    font-family: 'Noto Sans KR';
    font-size: 1.5rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    margin: 0 3.8125rem 0 3.5rem;
  `,
  dayRuleContainer: styled.div`
    display: flex;
    align-items: center;
    flex-direction: column;
    margin-top: 1.5rem;
    gap: 0.25rem;
  `,
  dayRule: styled.p`
    color: #000;
    text-align: center;
    font-family: 'Noto Sans KR';
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,
  person: styled.div`
    display: flex;
    width: 4.3125rem;
    padding: 0.375rem 0.75rem;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    border-radius: 26px;
    color: #fff;
    text-align: center;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    background: var(--Purple, #6b5ee1);
  `,

  rulesContainer: styled.div`
    display: flex;
    width: 64.5rem;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    gap: 1.375rem;
    margin: 0 3rem 3.0625rem 4.4375rem;
  `,
  rulesDescriptionContainer: styled.div`
    display: flex;
    height: 2.6875rem;
    justify-content: center;
    align-items: flex-start;
    gap: 51.6875rem;
  `,
  editButton: styled.button`
    display: flex;
    padding: 0.63rem 1.13rem;
    justify-content: center;
    align-items: center;
    border-radius: 16px;
    border: 1px solid var(--Main-1, #e15637);
    background: var(--White, #fff);
    color: var(--Main-1, #e15637);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,
  rulesContent: styled.div`
    width: 100%;
    height: 21.625rem;
    border-radius: 16px;
    background: #f7f6f9;
  `,

  accountContainer: styled.div`
    display: flex;
    width: 63.9375rem;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    gap: 1.375rem;
    margin: 0 3rem 9.375rem 4.4375rem;
  `,
  accountContent: styled.div`
    width: 64.5rem;
    height: 12.4375rem;
    border-radius: 16px;
    background: #f7f6f9;
  `,
};

interface UserProfileInfoProps {
  name: string | undefined;
  age: string;
  src: string | undefined;
}

function UserInfo({ name, age, src }: UserProfileInfoProps) {
  const [isChecked, setIsChecked] = useState(false);

  const toggleSwitch = () => {
    setIsChecked(!isChecked);
  };

  return (
    <styles.userProfileContainer>
      <styles.userProfileWithoutSwitch>
        <styles.userPicContainer>
          <styles.userPic src={src} alt="User Profile Pic" />
        </styles.userPicContainer>
        <styles.userInfoContainer>
          <styles.userName>{name}</styles.userName>
          <styles.userDetailedContainer>
            <styles.userDetailedInfo>{age}</styles.userDetailedInfo>
            <styles.userDetailedInfo>성북 길음동</styles.userDetailedInfo>
          </styles.userDetailedContainer>
        </styles.userInfoContainer>
      </styles.userProfileWithoutSwitch>
      <ToggleSwitch isChecked={isChecked} onToggle={toggleSwitch} />
    </styles.userProfileContainer>
  );
}

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
      <styles.switchDescription>메이트 찾는 중</styles.switchDescription>
    </styles.switchContainer>
  );
}

function Auth({ isMySelf }: { isMySelf: boolean }) {
  return (
    <styles.authContainer>
      <styles.authCheckImg src="/check_circle_24px copy.svg" />
      <styles.authDescription>본인인증</styles.authDescription>
    </styles.authContainer>
  );
}

function Card({
  name,
  memberId,
  myCardId,
  mateCardId,
  isMySelf,
}: {
  name: string | undefined;
  memberId: string | undefined;
  myCardId: number | undefined;
  mateCardId: number | undefined;
  isMySelf: boolean;
}) {
  return (
    <styles.cardSection>
      <styles.cardWrapper>
        <styles.description32px>내 카드</styles.description32px>
        <Link
          href={`/profile/card/${myCardId}?memberId=${memberId}&isMySelf=${isMySelf}`}
        >
          <styles.cardContainer>
            <styles.cardName>{name}</styles.cardName>
            <styles.cardDefault>기본</styles.cardDefault>
          </styles.cardContainer>
        </Link>
      </styles.cardWrapper>
      <styles.cardWrapper>
        <styles.description32px>메이트 카드</styles.description32px>
        <Link
          href={`/profile/card/${mateCardId}?memberId=${memberId}&isMySelf=${isMySelf}`}
        >
          <styles.cardContainer>
            <styles.cardName>메이트</styles.cardName>
            <styles.cardDefault>기본</styles.cardDefault>
          </styles.cardContainer>
        </Link>
      </styles.cardWrapper>
    </styles.cardSection>
  );
}

function Mon() {
  return (
    <styles.dayRuleContainer>
      <styles.dayRule>분리수거</styles.dayRule>
      <styles.person>김마루</styles.person>
    </styles.dayRuleContainer>
  );
}

function Wed() {
  return (
    <styles.dayRuleContainer>
      <styles.dayRule>화장실청소</styles.dayRule>
      <styles.person>김마루</styles.person>
    </styles.dayRuleContainer>
  );
}

function Sun() {
  return (
    <styles.dayRuleContainer>
      <styles.dayRule>방청소</styles.dayRule>
      <styles.person>김마루</styles.person>
    </styles.dayRuleContainer>
  );
}

function Maru() {
  return (
    <styles.maruContainer>
      <styles.description32px>마이 마루</styles.description32px>
      <styles.weekContainer>
        <styles.dayContainer>
          <styles.day>월</styles.day>
          <Mon />
        </styles.dayContainer>
        <styles.dayContainer>
          <styles.day>화</styles.day>
        </styles.dayContainer>
        <styles.dayContainer>
          <styles.day>수</styles.day>
          <Wed />
        </styles.dayContainer>
        <styles.dayContainer>
          <styles.day>목</styles.day>
        </styles.dayContainer>
        <styles.dayContainer>
          <styles.day>금</styles.day>
        </styles.dayContainer>
        <styles.dayContainer>
          <styles.day>토</styles.day>
        </styles.dayContainer>
        <styles.dayContainer>
          <styles.day>일</styles.day>
          <Sun />
        </styles.dayContainer>
      </styles.weekContainer>
      <styles.rulesContainer>
        <styles.rulesDescriptionContainer>
          <styles.description24px>생활 규칙</styles.description24px>
          <styles.editButton>수정하기</styles.editButton>
        </styles.rulesDescriptionContainer>
        <styles.rulesContent />
      </styles.rulesContainer>
      <styles.accountContainer>
        <styles.description24px>공용 계좌</styles.description24px>
        <styles.accountContent />
      </styles.accountContainer>
    </styles.maruContainer>
  );
}

interface UserProps {
  memberId: string;
  email: string;
  name: string;
  birthYear: string;
  gender: string;
  phoneNumber: string;
  initialized: boolean;
  myCardId: number;
  mateCardId: number;
}

export function ProfilePage({ memberId }: { memberId: string }) {
  const auth = useAuthValue();
  const { data } = useUserData(auth?.accessToken !== undefined);

  const id = data?.memberId;

  const user = useProfileData(memberId);
  const [userData, setUserData] = useState<UserProps | null>(null);
  const [isMySelf, setIsMySelf] = useState(false);

  useEffect(() => {
    if (user.data !== undefined) {
      const userProfileData = user.data.data.authResponse;
      const {
        name,
        email,
        birthYear,
        gender,
        phoneNumber,
        initialized,
        myCardId,
        mateCardId,
      } = userProfileData;
      setUserData({
        memberId,
        name,
        email,
        birthYear,
        gender,
        phoneNumber,
        initialized,
        myCardId,
        mateCardId,
      });
      if (id === memberId) {
        setIsMySelf(true);
      }
    }
  }, [user.data, memberId]);

  const birthYearString: string = userData?.birthYear ?? '';
  const birthYearDate: number = Number(birthYearString);
  return (
    <styles.pageContainer>
      <UserInfo
        name={userData?.name ?? ''}
        age={
          String(getAge(birthYearDate)) !== ''
            ? String(getAge(birthYearDate))
            : ''
        }
        src={user.data?.data.profileImage}
      />
      <Auth isMySelf={isMySelf} />
      <Card
        name={userData?.name}
        memberId={userData?.memberId}
        myCardId={userData?.myCardId}
        mateCardId={userData?.mateCardId}
        isMySelf={isMySelf}
      />
      <Maru />
    </styles.pageContainer>
  );
}
