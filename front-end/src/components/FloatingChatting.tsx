'use client';

import { Client } from '@stomp/stompjs';
import axios from 'axios';
import { useRouter } from 'next/navigation';
import React, { useEffect, useState } from 'react';
import { useRecoilState } from 'recoil';
import styled from 'styled-components';

import { ChattingList } from './chat/ChattingList';
import { ChattingRoom } from './chat/ChattingRoom';

import { useAuthValue, useUserData } from '@/features/auth';
import { chatOpenState, type GetChatRoomDTO } from '@/features/chat';
import { useIsMobile } from '@/shared/mobile';
import { useToast } from '@/features/toast';

const styles = {
  chattingButton: styled.div`
    position: fixed;
    bottom: 1rem;
    right: 1rem;
    display: inline-flex;
    padding: 1rem;
    align-items: flex-start;
    gap: 0.5rem;
    border-radius: 100px;
    background: #e15637;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    z-index: 3147483800;
    transition: transform 0.3s ease;
    cursor: pointer;

    &:hover {
      transform: scale(1.1);
    }
  `,
  buttonIcon: styled.img`
    width: 2rem;
    height: 2rem;

    @media (max-width: 768px) {
      width: 1.5rem;
      height: 1.5rem;
    }
  `,
  container: styled.div`
    position: fixed;
    bottom: 6rem;
    right: 6.5rem;
    display: flex;
    width: 25rem;
    height: 35rem;
    flex-direction: column;
    align-items: flex-start;
    flex-shrink: 0;
    border-radius: 20px;
    background: #fff;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    z-index: 1000;
    -webkit-transition: 0.4s;
    transition: 0.4s;
    overflow: hidden;
  `,
  chattingHeader: styled.div`
    display: inline-flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 1rem;
    gap: 1rem;
    width: 100%;
    height: 3.25rem;
    flex-shrink: 0;
    border-radius: 20px 20px 0 0;
    background: var(--background, #f7f6f9);
  `,
  title: styled.span`
    font-family: 'Pretendard';
    font-size: 1.575rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
  chattingSection: styled.div`
    width: 100%;
    height: calc(100% - 3.25rem);
    display: flex;
    flex-direction: column;
    overflow-y: scroll;

    &::-webkit-scrollbar {
      width: 0.5rem;
    }
    &::-webkit-scrollbar-thumb {
      background-color: #ced3da;
      border-radius: 4px;
    }
  `,
  searchButton: styled.img`
    width: 1.2rem;
    height: 1.2rem;
    cursor: pointer;
  `,
  searchInput: styled.input`
    flex: 1;
    font-size: 1.25rem;
    padding: 0.8rem;
    height: 2rem;
    width: 8rem;
    background-color: transparent;
    border: #bdbdbd solid 1px;
    border-radius: 1.2rem;
    color: var(--Text-grayDark, #2c2c2e);
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 400;
    line-height: 1.125rem;

    &:focus {
      outline: none;
    }
  `,
};

interface ChatRoom {
  roomId: number;
  roomName: string;
  unreadCount: number;
  lastMessage: string;
  lastMessageTime: string;
}

interface Message {
  createAt: string;
  message: string;
  messageId: string;
  nickname: string;
  roomId: number;
  sender: string;
}

function FloatingChattingBox() {
  const { createToast } = useToast();

  const [isChatRoomOpen, setIsChatRoomOpen] = useState<boolean>(false);
  const [chatRooms, setChatRooms] = useState<ChatRoom[]>([]);
  const [message, setMessage] = useState<Message>();
  const [selectedRoomId, setSelectedRoomId] = useState<number>(0);
  const [selectedRoomName, setSelectedRoomName] = useState<string>('');
  const [selectedRoomLastTime, setSelectedRoomLastTime] = useState<string>('');

  const auth = useAuthValue();
  const { data } = useUserData(auth?.accessToken !== undefined);
  const [userId, setUserId] = useState<string>('');
  const [userName, setUserName] = useState<string>('');

  const [stompClient, setStompClient] = useState<Client | null>(null);

  useEffect(() => {
    const initializeChat = async () => {
      try {
        const stomp = new Client({
          brokerURL: `${process.env.NEXT_PUBLIC_CHAT_API_URL}`,
          connectHeaders: {
            Authorization: `Bearer ${auth?.accessToken}`,
          },
          debug: (str: string) => {
            console.log(str);
          },
          reconnectDelay: 5000,
          heartbeatIncoming: 4000,
          heartbeatOutgoing: 4000,
        });
        setStompClient(stomp);

        stomp.onConnect = () => {
          createToast({
            message: '채팅 연결에 성공했습니다.',
            option: {
              duration: 3000,
            },
          });

          stomp.subscribe(`/roomList/${userId}`, frame => {
            try {
              const newMessage: Message = JSON.parse(frame.body);
              if (newMessage != null) setMessage(newMessage);
            } catch (error) {
              console.error('오류가 발생했습니다:', error);
            }
          });
        };

        stomp.onWebSocketError = () => {
          createToast({
            message: '연결에 실패했습니다. 다시 시도합니다.',
            option: {
              duration: 3000,
            },
          });
        };

        stomp.activate();
      } catch (error) {
        console.error('채팅 방 생성 중 오류가 발생했습니다:', error);
      }
    };

    if (auth?.accessToken != null) {
      initializeChat();
    }

    return () => {
      if (stompClient != null && stompClient.connected) {
        stompClient.deactivate();
      }
    };
  }, [auth?.accessToken]);

  useEffect(() => {
    if (data !== undefined) {
      setUserId(data.memberId);
      setUserName(data.name);
    }
  }, [data]);

  const handleChatRoomClick = () => {
    setIsChatRoomOpen(true);
  };

  useEffect(() => {
    setTimeout(() => {
      (async () => {
        if (!isChatRoomOpen) {
          try {
            const res = await axios.get<GetChatRoomDTO>('/maru-api/chatRoom');
            const chatRoomListData: ChatRoom[] = res.data.data;
            setChatRooms(chatRoomListData);
            return true;
          } catch (error) {
            console.error(error);
            return false;
          }
        }
        return true;
      })();
    }, 50);
  }, [auth, isChatRoomOpen]);

  useEffect(() => {
    if (!isChatRoomOpen) {
      setChatRooms(prevChatRooms =>
        prevChatRooms.map(room => {
          if (room.roomId === message?.roomId) {
            return {
              ...room,
              unreadCount: room.unreadCount + 1,
              lastMessage: message.message,
            };
          }
          return room;
        }),
      );
    }
  }, [message, isChatRoomOpen]);

  return (
    <>
      <styles.container>
        <styles.chattingHeader>
          <div>
            <styles.title style={{ color: 'var(--Main-1, #E15637)' }}>
              maru{' '}
            </styles.title>
            <styles.title style={{ color: 'var(--Gray, #8C95A8)' }}>
              chat
            </styles.title>
          </div>
        </styles.chattingHeader>

        <styles.chattingSection>
          {chatRooms.map((room, index) => (
            <ChattingList
              key={index}
              name={room.roomName}
              unreadCount={room.unreadCount}
              lastMessage={room.lastMessage}
              onClick={() => {
                handleChatRoomClick();
                setSelectedRoomId(room.roomId);
                setSelectedRoomName(room.roomName);
                setSelectedRoomLastTime(room.lastMessageTime);
              }}
            />
          ))}
        </styles.chattingSection>
      </styles.container>
      {isChatRoomOpen && (
        <ChattingRoom
          userId={userId}
          userName={userName}
          roomId={selectedRoomId}
          roomName={selectedRoomName}
          lastTime={selectedRoomLastTime}
          onRoomClick={setIsChatRoomOpen}
        />
      )}
    </>
  );
}

export function FloatingChatting() {
  const [isChatOpen, setIsChatOpen] = useRecoilState(chatOpenState);
  const router = useRouter();

  const toggleChat = () => {
    setIsChatOpen(prevState => !prevState);
  };

  const isMobile = useIsMobile();
  useEffect(() => {
    if (isChatOpen && isMobile) {
      router.replace('/chat');
    }
    if (!isChatOpen && isMobile) {
      router.replace('/');
    }
  }, [isChatOpen, isMobile]);

  useEffect(() => {
    if (!isMobile && window.location.pathname === '/chat') {
      router.replace('/');
    }
  }, [isMobile]);

  const auth = useAuthValue();
  if (auth == null) return <></>;
  return (
    <>
      <styles.chattingButton onClick={toggleChat}>
        <styles.buttonIcon src="/chatting.svg" />
      </styles.chattingButton>
      {isChatOpen && !isMobile ? <FloatingChattingBox /> : null}
    </>
  );
}
