'use client';

import Link from 'next/link';
import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

import { Bookmark } from '@/components';
import { useAuthValue, useUserData } from '@/features/auth';
import {
  useFollowUser,
  useFollowingListData,
  useUnfollowUser,
  useUserProfile,
} from '@/features/profile';

const styles = {
  container: styled.div`
    display: flex;
    width: 100vw;
    min-width: 390px;
    padding-bottom: 2rem;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem;
    gap: 3rem;
  `,

  userProfileContainer: styled.div`
    display: flex;
    align-items: flex-start;
    gap: 1.25rem;
    align-self: stretch;
  `,
  userProfileWithoutInfo: styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
  `,
  userPicContainer: styled.div`
    display: flex;
    width: 6.0625rem;
    height: 6.0625rem;
    justify-content: center;
    align-items: center;
    border-radius: 50%;
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
    padding: 0.5rem 0rem;
    flex-direction: column;
    justify-content: space-between;
    align-items: flex-start;
    flex: 1 0 0;
    align-self: stretch;
  `,
  userDetailedContainer: styled.div`
    display: inline-flex;
    width: 100%;
    align-items: flex-start;
    gap: 2rem;
  `,
  userName: styled.div`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,
  userDetailedInfo: styled.p`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,

  switchContainer: styled.div`
    display: inline-flex;
    justify-content: center;
    align-items: center;
    gap: 0.375rem;
  `,
  switchWrapper: styled.label`
    position: relative;
    display: inline-block;
    width: 2.2rem;
    height: 1.26rem;
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
    border-radius: 20px;
  `,
  sliderDot: styled.span`
    position: absolute;
    cursor: pointer;
    top: 0.1624rem;
    left: 0.2rem;
    background-color: white;
    -webkit-transition: 0.4s;
    transition: 0.4s;
    border-radius: 50%;
    width: 0.9rem;
    height: 0.9rem;
  `,
  switchDescription: styled.p`
    color: var(--Gray-3, #888);
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,

  authContainer: styled.div`
    height: 2rem;
    width: 5.3125rem;
    border-radius: 26px;
    background: var(--Black, #35373a);
    cursor: pointer;
    display: inline-flex;
    padding: 0.125rem 0.5rem;
    justify-content: center;
    align-items: center;
    gap: 0.25rem;
  `,
  authDescription: styled.p`
    color: #fff;

    font-family: 'Noto Sans KR';
    font-size: 0.75rem;
    font-style: normal;
    font-weight: 400;
    line-height: 1.5rem; /* 200% */
  `,
  authCheckImg: styled.img`
    width: 1rem;
    height: 1rem;
  `,

  cardSection: styled.div`
    display: flex;
    width: 100%;
    align-items: flex-start;
    gap: 1.5rem;
    align-self: stretch;
  `,
  cardWrapper: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  `,
  description32px: styled.p`
    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
  cardContainer: styled.div`
    display: inline-flex;
    align-items: center;
    width: 9.8125rem;
    height: 9.8125rem;
    padding: 3.9375rem 5.8125rem 4.4375rem 1.1875rem;
    flex-shrink: 0;
    border-radius: 20px;
    border: 1pxs olid var(--background, #f7f6f9);
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    background: var(--grey-100, #fff);
    position: relative;
  `,
  cardName: styled.p`
    color: var(--grey-900, #494949);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
};

interface UserProfileInfoProps {
  name: string | undefined;
  email: string | undefined;
  phoneNum: string | undefined;
  src: string | undefined;
  memberId: string;
  isMySelf: boolean;
}

function UserInfo({
  name,
  email,
  phoneNum,
  src,
  memberId,
  isMySelf,
}: UserProfileInfoProps) {
  const [isChecked, setIsChecked] = useState(false);

  const followList = useFollowingListData();
  const [isMarked, setIsMarked] = useState(
    followList.data?.data.followingList[memberId] != null,
  );

  const toggleSwitch = () => {
    setIsChecked(!isChecked);
  };

  const { mutate: follow } = useFollowUser(memberId);
  const { mutate: unfollow } = useUnfollowUser(memberId);

  return (
    <styles.userProfileContainer>
      <styles.userProfileWithoutInfo>
        <styles.userPicContainer>
          <styles.userPic src={src} alt="User Profile Pic" />
        </styles.userPicContainer>
        <Auth />
      </styles.userProfileWithoutInfo>
      <styles.userInfoContainer>
        <styles.userName>{name}</styles.userName>
        <styles.userDetailedContainer>
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.25rem',
            }}
          >
            <styles.userDetailedInfo>{phoneNum}</styles.userDetailedInfo>
            <styles.userDetailedInfo>{email}</styles.userDetailedInfo>
          </div>
          {!isMySelf && (
            <Bookmark
              marked={isMarked}
              onToggle={() => {
                if (isMarked) unfollow();
                else follow();
                setIsMarked(prev => !prev);
              }}
              hasBorder
              color="#888"
            />
          )}
        </styles.userDetailedContainer>
        <ToggleSwitch isChecked={isChecked} onToggle={toggleSwitch} />
      </styles.userInfoContainer>
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
              transform: isChecked ? 'translateX(0.9rem)' : 'translateX(0)',
            }}
          />
        </styles.slider>
      </styles.switchWrapper>
      <styles.switchDescription>메이트 찾는 중</styles.switchDescription>
    </styles.switchContainer>
  );
}

function Auth() {
  return (
    <styles.authContainer>
      <styles.authCheckImg src="/check_circle_24px copy.svg" />
      <styles.authDescription>학교인증</styles.authDescription>
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
        <styles.description32px>마이 카드</styles.description32px>
        <Link
          href={`/profile/card/${myCardId}?memberId=${memberId}&isMySelf=${isMySelf}&type=myCard`}
        >
          <styles.cardContainer>
            <styles.cardName>{name}</styles.cardName>
          </styles.cardContainer>
        </Link>
      </styles.cardWrapper>
      <styles.cardWrapper>
        <styles.description32px>메이트 카드</styles.description32px>
        <Link
          href={`/profile/card/${mateCardId}?memberId=${memberId}&isMySelf=${isMySelf}&type=mateCard`}
        >
          <styles.cardContainer>
            <styles.cardName>메이트</styles.cardName>
          </styles.cardContainer>
        </Link>
      </styles.cardWrapper>
    </styles.cardSection>
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

export function MobileProfilePage({ memberId }: { memberId: string }) {
  const auth = useAuthValue();
  const { data } = useUserData(auth?.accessToken !== undefined);

  const authId = data?.memberId;

  const [userData, setUserData] = useState<UserProps | null>(null);
  const [isMySelf, setIsMySelf] = useState(false);

  const { mutate: mutateProfile, data: profileData } = useUserProfile(memberId);
  const [profileImg, setProfileImg] = useState<string>('');

  useEffect(() => {
    mutateProfile();
  }, [auth]);

  useEffect(() => {
    if (profileData?.data !== undefined) {
      const userProfileData = profileData.data.authResponse;
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
      setProfileImg(profileData.data.profileImage);
      if (authId === memberId) {
        setIsMySelf(true);
      }
    }
  }, [profileData, memberId]);

  return (
    <styles.container>
      <UserInfo
        name={userData?.name ?? ''}
        email={userData?.email ?? ''}
        phoneNum={userData?.phoneNumber ?? ''}
        src={profileImg}
        memberId={memberId}
        isMySelf={isMySelf}
      />
      <Card
        name={userData?.name}
        memberId={userData?.memberId}
        myCardId={userData?.myCardId}
        mateCardId={userData?.mateCardId}
        isMySelf={isMySelf}
      />
    </styles.container>
  );
}
