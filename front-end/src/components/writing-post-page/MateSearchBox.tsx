'use client';

import { useEffect, useRef } from 'react';
import styled from 'styled-components';

import { useAuthValue } from '@/features/auth';
import { useMutualFollowUsers } from '@/features/profile';

const styles = {
  container: styled.div`
    position: fixed;
    width: 50dvh;
    max-width: 512px;
    height: 50dvh;
    max-height: 512px;
    background-color: white;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    border-radius: 16px;
    padding: 2rem;

    top: 50%;
    left: 50%;
    translate: -50% -50%;

    display: flex;
    flex-direction: column;
    gap: 1rem;

    @media (max-width: 768px) {
      padding: 1rem;
      width: 60dvw;
    }
  `,
  row: styled.div`
    display: flex;
    flex-direction: row;
    gap: 0.5rem;
    align-items: end;

    @media (max-width: 768px) {
      flex-wrap: wrap;
    }
  `,
  title: styled.h1`
    font-size: 1.25rem;

    @media (max-width: 768px) {
      font-size: 1rem;
    }
  `,
  caption: styled.span`
    font-size: 0.85rem;
    color: #808080;
  `,
  userList: styled.ol`
    display: flex;
    flex-direction: column;
    gap: 1rem;

    overflow-y: auto;
    overscroll-behavior-y: none;
  `,
  userItem: styled.li`
    cursor: pointer;

    display: flex;
    justify-content: space-between;
    gap: 1rem;

    div {
      display: flex;
      align-items: center;
      gap: 1rem;

      img {
        width: 50px;
        height: 50px;

        object-fit: cover;
        object-position: center;
        border-radius: 50%;
      }
    }

    padding-bottom: 8px;
    border-bottom: 0.5px solid #80808080;

    transition: 200ms scale ease-out;
    &:hover {
      scale: 1.005;
    }
  `,
};

function UserItem({
  memberId,
  nickname,
  profileImage,
  isSelected,
  onMateSelected,
}: {
  memberId: string;
  nickname: string;
  profileImage: string;
  isSelected: boolean;
  onMateSelected: (member: {
    memberId: string;
    nickname: string;
    profileImage: string;
  }) => void;
}) {
  return (
    <styles.userItem
      onClick={() => {
        onMateSelected({ memberId, nickname, profileImage });
      }}
    >
      <div>
        <img alt="Profile" src={profileImage} />
        <p>{nickname}</p>
      </div>
      {isSelected && <img alt="check" src="/icon-check.svg" />}
    </styles.userItem>
  );
}

export function MateSearchBox({
  selectedMates,
  setHidden,
  onMateSelected,
}: {
  selectedMates: Set<string>;
  setHidden: () => void;
  onMateSelected: (member: {
    memberId: string;
    nickname: string;
    profileImage: string;
  }) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  const auth = useAuthValue();
  const { data: users } = useMutualFollowUsers(auth?.user != null);

  useEffect(() => {
    const clickHandler = (event: MouseEvent) => {
      if (
        containerRef.current !== null &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setHidden();
      }
    };

    const keyHandler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setHidden();
    };

    document.addEventListener('click', clickHandler);
    document.addEventListener('keydown', keyHandler);
    return () => {
      document.removeEventListener('click', clickHandler);
      document.removeEventListener('keydown', keyHandler);
    };
  }, []);

  return (
    <styles.container ref={containerRef}>
      <styles.row>
        <styles.title>메이트 찾기</styles.title>
        <styles.caption>(서로 팔로우된 경우에만 보입니다.)</styles.caption>
      </styles.row>
      <styles.userList>
        {users?.map(({ memberId, nickname, profileImage }) => (
          <UserItem
            key={memberId}
            onMateSelected={onMateSelected}
            memberId={memberId}
            nickname={nickname}
            profileImage={profileImage}
            isSelected={selectedMates.has(memberId)}
          />
        ))}
      </styles.userList>
    </styles.container>
  );
}
