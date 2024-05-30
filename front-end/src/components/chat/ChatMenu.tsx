'use client';

import { useQueryClient } from '@tanstack/react-query';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { useRecoilState } from 'recoil';
import styled from 'styled-components';

import { useAuthValue } from '@/features/auth';
import {
  chatOpenState,
  useChangeChatRoomName,
  useChatRoomUser,
  useDeleteChatRoom,
  useInviteUsers,
} from '@/features/chat';
import { useSearchUser } from '@/features/profile';
import { useToast } from '@/features/toast';

const styles = {
  menuContainer: styled.div`
    position: absolute;
    top: 0;
    right: 0;
    display: inline-flex;
    flex-direction: column;
    width: 18rem;
    height: 100%;
    border-radius: 20px;
    background: #fff;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    z-index: 20;

    @media (max-width: 768px) {
      border-radius: 0;
    }
  `,
  header: styled.div`
    width: 100%;
    height: 3.625rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 0.8rem;
    box-shadow: 0px -1px 0px 0px #e5e5ea inset;
  `,
  closeButton: styled.img`
    width: 1.2rem;
    height: 1.2rem;
    border: none;
    flex-shrink: 0;
    cursor: pointer;
  `,
  menuListContainer: styled.ul`
    display: flex;
    flex: 1;
    flex-direction: column;
    padding: 0.8rem 0.8rem;
    width: 100%;
    box-shadow: 0px -1px 0px 0px #e5e5ea inset;
  `,
  footer: styled.div`
    width: 100%;
    height: 3rem;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0 0.8rem;
  `,
  menuList: styled.li`
    width: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 0.8rem 0.8rem;
    gap: 0.8rem;
    border-bottom: #e5e5ea solid 1px;
    cursor: pointer;

    color: var(--Text-grayDark, #2c2c2e);
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 400;
    line-height: 1.125rem; /* 128.571% */

    position: relative;
  `,
  inviteButton: styled.button`
    border: none;
    background-color: #fff;
    color: var(--Text-grayDark, #2c2c2e);
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 400;
    line-height: 1.125rem; /* 128.571% */
    text-align: start;
    cursor: pointer;
  `,
  userListContainer: styled.ul`
    display: flex;
    flex: 1;
    flex-direction: column;
    padding: 0 0.8rem;
    width: 100%;
    max-height: 12rem;
    overflow-y: auto;
    gap: 1rem;

    &::-webkit-scrollbar {
      width: 0.5rem;
    }
    &::-webkit-scrollbar-thumb {
      background-color: #ced3da;
      border-radius: 4px;
    }
  `,
  userList: styled.li`
    display: flex;
    gap: 0.4rem;
    align-items: center;
    list-style: none;
    color: var(--Text-gray, #666668);
    cursor: pointer;
  `,
  userImg: styled.img`
    width: 1.5rem;
    height: 1.5rem;
    flex-shrink: 0;
    border-radius: 150px;
    border: 1.5px solid #fff;
  `,
  searchInput: styled.input`
    flex: 1;
    font-size: 1.25rem;
    padding: 0 0.8rem;
    border: none;
    color: var(--Text-grayDark, #2c2c2e);
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;

    &:focus {
      outline: none;
    }
  `,
  searchButton: styled.img`
    width: 1.2rem;
    height: 1.2rem;
    cursor: pointer;
  `,
  dropDownContainer: styled.div`
    width: 100%;
    overflow: hidden;
    z-index: 1;
  `,
  followingListContainer: styled.div`
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.625rem;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    background: #fff;
    border: 1px solid #e5e5ea;
    @keyframes dropdown {
      0% {
        transform: translateY(-100%);
      }
      100% {
        transform: translateY(0);
      }
    }
    animation: dropdown 0.5s ease;
  `,
  followingUserContainer: styled.ul`
    display: flex;
    padding: 0.625rem;
    align-items: center;
    gap: 0.625rem;
    align-self: stretch;
  `,
  searchBox: styled.div`
    display: flex;
    height: 1.6875rem;
    justify-content: space-between;
    align-items: center;
    flex: 1 0 0;
    border-radius: 1.25rem;
    border: 1px solid #888;
    padding: 0.1rem 0.8rem;
  `,
};

interface User {
  memberId: string;
  nickname: string;
  profileImageUrl: string;
}

export function ChatMenu({
  roomId,
  roomName,
  onMenuClicked,
}: {
  roomId: number;
  roomName: string;
  onMenuClicked: React.Dispatch<React.SetStateAction<boolean>>;
}) {
  const [isCloseClick, setIsCloseClick] = useState<boolean>(false);
  const [isInviteClick, setIsInviteClick] = useState<boolean>(false);
  const users = useChatRoomUser(roomId);
  const [userList, setUserList] = useState<User[]>([]);
  const [, setIsChatOpen] = useRecoilState(chatOpenState);

  const auth = useAuthValue();

  useEffect(() => {
    if (users.data !== undefined) {
      const userListData: User[] = users.data.data;
      setUserList(userListData);
    }
  }, [users.data]);

  const handleCloseClick = () => {
    setIsCloseClick(prev => !prev);
    onMenuClicked(isCloseClick);
  };

  const [email, setEmail] = useState<string>('');
  const {
    mutate: mutateSearchUser,
    data: searchData,
    error,
  } = useSearchUser(email);

  const [searchUser, setSearchUser] = useState<User>();

  useEffect(() => {
    if (searchData?.data != null) {
      setSearchUser(searchData.data);
    }
  }, [searchData]);

  const { createToast } = useToast();

  useEffect(() => {
    if (error != null) {
      createToast({
        message: '존재하지 않는 유저입니다.',
        option: {
          duration: 3000,
        },
      });
    }
  }, [error]);

  const [deleteName, setDeleteName] = useState('');

  const createDeleteChatRoomName = () => {
    let newName = '';
    userList.map((user, index) => {
      if (user.nickname !== auth?.user?.name) {
        if (index !== 0) newName = `${newName}, ${user.nickname}`;
        else newName = user.nickname;
      }
      return true;
    });
    setDeleteName(newName);
  };

  const { mutate: inviteUser } = useInviteUsers(roomId, [
    searchUser?.memberId ?? '',
  ]);
  const { mutate: setInviteChatRoomName } = useChangeChatRoomName(
    `${roomName}, ${searchUser?.nickname}`,
    roomId,
  );
  const { mutate: setDeleteChatRoomName } = useChangeChatRoomName(
    `${deleteName}`,
    roomId,
  );

  const { mutate: deleteChatRoom } = useDeleteChatRoom();

  function handleKeyUp(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.keyCode === 13) {
      mutateSearchUser();
    }
  }

  const queryClient = useQueryClient();

  return (
    <styles.menuContainer>
      <styles.header>
        <styles.closeButton src="/Close.svg" onClick={handleCloseClick} />
      </styles.header>
      <styles.menuListContainer>
        <styles.menuList>
          하우스 메이트
          <styles.userListContainer>
            {userList.map((user, index) => (
              <Link href={`/profile/${user.memberId}`} key={index}>
                <styles.userList key={index}>
                  <styles.userImg src={user.profileImageUrl} />
                  {user.nickname}
                </styles.userList>
              </Link>
            ))}
          </styles.userListContainer>
        </styles.menuList>
        <styles.menuList>
          <styles.inviteButton
            onClick={() => {
              setIsInviteClick(prev => !prev);
            }}
          >
            메이트 초대하기
          </styles.inviteButton>
          {isInviteClick && (
            <styles.dropDownContainer>
              <styles.followingListContainer>
                <styles.searchBox>
                  <styles.searchInput
                    style={{ width: '100%' }}
                    onChange={e => {
                      setEmail(e.target.value);
                    }}
                    onKeyUp={handleKeyUp}
                  />
                  <styles.searchButton src="/icon-search.svg" />
                </styles.searchBox>
                <styles.followingUserContainer>
                  {searchUser != null ? (
                    <styles.userList
                      onClick={() => {
                        inviteUser(undefined, {
                          onSuccess: () => {
                            queryClient.invalidateQueries({
                              queryKey: [`/chatRoom/${roomId}`],
                            });
                          },
                        });
                        setInviteChatRoomName();
                      }}
                    >
                      <styles.userImg src={searchUser?.profileImageUrl} />
                      {searchUser?.nickname}
                    </styles.userList>
                  ) : null}
                </styles.followingUserContainer>
              </styles.followingListContainer>
            </styles.dropDownContainer>
          )}
        </styles.menuList>
        <styles.menuList
          onClick={() => {
            deleteChatRoom(roomId);
            createDeleteChatRoomName();
            setDeleteChatRoomName();
            setIsChatOpen(false);
          }}
        >
          채팅방 나가기
        </styles.menuList>
      </styles.menuListContainer>
    </styles.menuContainer>
  );
}
