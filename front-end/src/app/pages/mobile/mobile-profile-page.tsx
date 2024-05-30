'use client';

import axios from 'axios';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import React, { useState, useEffect, useRef } from 'react';
import { atom, useRecoilState, useRecoilValue } from 'recoil';
import styled from 'styled-components';

import { Bookmark } from '@/components';
import { useAuthValue, useUserData } from '@/features/auth';
import { chatOpenState, useCreateChatRoom } from '@/features/chat';
import { getImageURL, putImage } from '@/features/image';
import {
  type GetFollowingListDTO,
  useCertification,
  useFollowUser,
  useGetCode,
  useUnfollowUser,
  useUserProfile,
  useProfileSetting,
} from '@/features/profile';
import { convertPhoneNumber } from '@/shared';

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
    gap: 1rem;
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

  certificationContainer: styled.div`
    display: inline-flex;
    padding: 1.5rem;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    width: 22rem;
    height: 15rem;
    gap: 1.5rem;
    border-radius: 20px;
    background: #fff;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.25);
    z-index: 20000;
    position: absolute;
    top: 18rem;
    left: 0.5rem;

    p {
      color: #494949;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;
    }
  `,
  userInputList: styled.ul`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    gap: 0.4rem;
  `,
  userInputListItem: styled.li`
    display: flex;
    align-items: center;
    gap: 0.5rem;

    div {
      display: flex;
      width: 12rem;
      justify-content: space-between;
      align-items: center;
    }

    p {
      color: #494949;
      font-family: 'Noto Sans KR';
      font-size: 0.875rem;
      font-style: normal;
      font-weight: 400;
      line-height: normal;
    }

    input {
      width: 8rem;
      height: 2rem;
      flex-shrink: 0;
      border-radius: 12px;
      border: 1px solid #494949;
      padding: 0.5rem 1rem;
    }
  `,

  certificationButton: styled.button`
    display: flex;
    width: 7rem;
    padding: 0.3rem 0.8rem;
    justify-content: center;
    align-items: center;
    gap: 0.25rem;
    border-radius: 8px;
    background-color: white;
    border: 1px solid #494949;

    color: #494949;
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
  `,

  chattingButton: styled.button`
    all: unset;
    cursor: pointer;

    display: flex;
    padding: 0.5rem 1.5rem;
    justify-content: center;
    align-items: center;
    gap: 0.25rem;
    align-self: stretch;

    border-radius: 8px;
    background: var(--Black, #35373a);

    color: #fff;
    font-family: Pretendard;
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 600;
    line-height: 1.5rem;
  `,

  postContainer: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    align-self: stretch;

    h1 {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1.25rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
    }
  `,

  posts: styled.div`
    display: flex;
    padding: 2rem;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    align-self: stretch;
    border-radius: 20px;
    background: #fff;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.25);
  `,

  postName: styled.p`
    color: #000;

    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;

    cursor: pointer;
  `,
};

interface UserProfileInfoProps {
  name: string | undefined;
  email: string | undefined;
  phoneNum: string | undefined;
  src: string | undefined;
  memberId: string;
  isMySelf: boolean;
  certification?: boolean;
  myID: string;
  myName: string;
  recommendOn: boolean;
  onProfileImageChanged: () => void;
}

function UserInfo({
  name,
  email,
  phoneNum,
  src,
  memberId,
  isMySelf,
  certification,
  myID,
  myName,
  recommendOn,
  onProfileImageChanged,
}: UserProfileInfoProps) {
  const [isChecked, setIsChecked] = useState(recommendOn);

  useEffect(() => {
    setIsChecked(recommendOn);
  }, [recommendOn]);

  const [followList, setFollowList] = useState<
    Array<{
      memberId: string;
      nickname: string;
      profileImage: string;
    }>
  >();
  const [isMarked, setIsMarked] = useState(false);

  useEffect(() => {
    if (followList?.find(elem => elem.memberId === memberId) != null) {
      setIsMarked(true);
    } else setIsMarked(false);
  }, [followList]);

  useEffect(() => {
    (async () => {
      try {
        const res = await axios.get<GetFollowingListDTO>(
          '/maru-api/profile/follow',
        );
        const followListData = res.data.data.followingList;
        setFollowList(followListData);
        return true;
      } catch (error) {
        console.error(error);
        return false;
      }
    })();
  }, [setIsMarked]);

  const { mutate: settingRecommend } = useProfileSetting();

  const toggleSwitch = () => {
    settingRecommend(!isChecked);
    setIsChecked(!isChecked);
  };

  const { mutate: follow } = useFollowUser(memberId);
  const { mutate: unfollow } = useUnfollowUser(memberId);

  const [, setIsChatOpen] = useRecoilState(chatOpenState);

  const { mutate: chattingMutate } = useCreateChatRoom();

  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const handleImageInputClicked = () => {
    if (isMySelf) imageInputRef.current?.click();
  };

  const changeProfileImage = async (file: File) => {
    try {
      const result = await getImageURL(`.${file.type.split('/')[1]}`);

      await putImage(result.data.data.url, file);

      await axios.patch('/maru-api/profile/image', result.data.data.fileName, {
        headers: {
          'Content-Type': 'text/plain',
        },
      });
    } catch (error) {
      console.error(error);
    }
    onProfileImageChanged();
  };

  return (
    <styles.userProfileContainer>
      <styles.userProfileWithoutInfo>
        <styles.userPicContainer onClick={handleImageInputClicked}>
          <input
            type="file"
            ref={imageInputRef}
            onChange={e => {
              const file = e.target.files?.[0];
              if (file != null) {
                changeProfileImage(file);
              }
            }}
            style={{ display: 'none' }}
          />
          <styles.userPic src={src} alt="User Profile Pic" />
        </styles.userPicContainer>
        <Auth certification={certification} isMySelf={isMySelf} />
      </styles.userProfileWithoutInfo>
      <styles.userInfoContainer>
        <styles.userName>{name}</styles.userName>
        <ToggleSwitch
          isChecked={isChecked}
          onToggle={toggleSwitch}
          isMySelf={isMySelf}
        />
        <styles.userDetailedContainer>
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.25rem',
            }}
          >
            <styles.userDetailedInfo>
              {phoneNum != null && convertPhoneNumber(phoneNum)}
            </styles.userDetailedInfo>
            <styles.userDetailedInfo>{email}</styles.userDetailedInfo>
          </div>
          {!isMySelf && (
            <>
              <styles.chattingButton
                onClick={() => {
                  if (name != null)
                    chattingMutate({
                      roomName: `${name}, ${myName}`,
                      members: [memberId],
                      myID,
                    });

                  setTimeout(() => {
                    setIsChatOpen(true);
                  }, 200);
                }}
              >
                채팅
              </styles.chattingButton>
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
            </>
          )}
        </styles.userDetailedContainer>
      </styles.userInfoContainer>
    </styles.userProfileContainer>
  );
}

interface ToggleSwitchProps {
  isChecked: boolean;
  onToggle: () => void;
  isMySelf: boolean;
}

function ToggleSwitch({ isChecked, onToggle, isMySelf }: ToggleSwitchProps) {
  return (
    <styles.switchContainer>
      <styles.switchWrapper>
        <styles.switchInput
          type="checkbox"
          checked={isChecked}
          onChange={() => {
            if (isMySelf) onToggle();
          }}
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

function Auth({
  certification,
  isMySelf,
}: {
  certification?: boolean;
  isMySelf: boolean;
}) {
  const [univName, setUnivName] = useState<string>();
  const [email, setEmail] = useState<string>();
  const [code, setCode] = useState<number>();
  const [isCertification, setIsCertification] = useState(certification);

  useEffect(() => {
    setIsCertification(certification);
  }, [certification]);

  const { mutate: getCode } = useGetCode(email ?? '', univName ?? '');
  const { mutate: postCertification, data: success } = useCertification(
    email ?? '',
    univName ?? '',
    code ?? 0,
  );

  useEffect(() => {
    setIsCertification(true);
  }, [success]);

  const [isCertificationClick, setIsCertificationClick] = useState(false);
  return (
    <>
      <styles.authContainer
        onClick={() => {
          if (isMySelf) setIsCertificationClick(prev => !prev);
        }}
      >
        <styles.authCheckImg
          src={
            isCertification != null && isCertification
              ? '/check_circle_24px copy.svg'
              : '/Close_white.svg'
          }
        />
        <styles.authDescription>학교인증</styles.authDescription>
      </styles.authContainer>
      {isCertificationClick && (
        <styles.certificationContainer>
          <p>학교 인증하기</p>
          <styles.userInputList>
            <styles.userInputListItem>
              <div>
                <p>대학교 명</p>
                <input
                  onChange={e => {
                    setUnivName(e.target.value);
                  }}
                />
              </div>
            </styles.userInputListItem>
            <styles.userInputListItem>
              <div>
                <p>이메일</p>
                <input
                  onChange={e => {
                    setEmail(e.target.value);
                  }}
                />
              </div>
              <styles.certificationButton
                onClick={() => {
                  getCode();
                }}
              >
                인증코드 받기
              </styles.certificationButton>
            </styles.userInputListItem>
            <styles.userInputListItem>
              <div>
                <p>인증코드</p>
                <input
                  onChange={e => {
                    setCode(Number(e.target.value));
                  }}
                />
              </div>
              <styles.certificationButton
                style={{
                  color: '#fff',
                  backgroundColor: ' #E15637',
                  border: 'none',
                }}
                onClick={() => {
                  postCertification();
                }}
              >
                인증하기
              </styles.certificationButton>
            </styles.userInputListItem>
          </styles.userInputList>
        </styles.certificationContainer>
      )}
    </>
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

function Posts({ posts }: { posts?: PostsProps[] }) {
  return (
    <styles.postContainer>
      <h1>게시글</h1>
      <styles.posts>
        {posts?.length === 0 && (
          <styles.postName>게시글이 없습니다.</styles.postName>
        )}
        {posts?.map(post => (
          <Link
            key={post.id}
            href={`/shared/${post.type === 'studio' ? 'room' : 'dormitory'}/${post.id}`}
          >
            <styles.postName>{post.title}</styles.postName>
          </Link>
        ))}
      </styles.posts>
    </styles.postContainer>
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
  univCertified: boolean;
}

interface PostsProps {
  id: number;
  title: string;
  type: string;
  createdAt: string;
  modifiedAt: string;
}

export function MobileProfilePage({ memberId }: { memberId: string }) {
  const auth = useAuthValue();
  const { data } = useUserData(auth?.accessToken !== undefined);
  const router = useRouter();

  const authId = data?.memberId;

  const [userData, setUserData] = useState<UserProps | null>(null);
  const [isMySelf, setIsMySelf] = useState(false);

  const {
    mutate: mutateProfile,
    data: profileData,
    error,
  } = useUserProfile(memberId);
  const [profileImg, setProfileImg] = useState<string>('');
  const [posts, setPosts] = useState<PostsProps[]>();
  const [recommendOn, setRecommendOn] = useState(false);

  useEffect(() => {
    if (error != null) router.replace('/error');
  }, [error]);

  useEffect(() => {
    mutateProfile();
  }, [auth, mutateProfile]);

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
        univCertified,
        mateCardId,
        myCardId,
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
        univCertified,
      });
      setProfileImg(profileData.data.profileImage);
      setPosts(profileData.data.posts);
      setRecommendOn(profileData.data.recommendOn);
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
        certification={userData?.univCertified}
        myID={authId ?? ''}
        myName={auth?.user?.name ?? ''}
        recommendOn={recommendOn}
        onProfileImageChanged={() => {
          mutateProfile();
        }}
      />
      <Card
        name={userData?.name}
        memberId={userData?.memberId}
        myCardId={userData?.myCardId}
        mateCardId={userData?.mateCardId}
        isMySelf={isMySelf}
      />
      <Posts posts={posts} />
    </styles.container>
  );
}
