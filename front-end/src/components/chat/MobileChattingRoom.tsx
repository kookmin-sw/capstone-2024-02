'use client';

import { Client } from '@stomp/stompjs';
import { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';

import { ChatMenu } from './ChatMenu';
import { ReceiverMessage } from './ReceiverMessage';
import { SenderMessage } from './SenderMessage';

import { useAuthValue } from '@/features/auth';
import { useEnterChatRoom, useExitChatRoom } from '@/features/chat';

const styles = {
  container: styled.div`
    display: flex;
    width: 100vw;
    height: 100%;
    min-width: 390px;
    padding-bottom: 2rem;
    flex-direction: column;
    align-items: center;
    position: absolute;
    top: 0;
    background-color: white;
    z-index: 30000;
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
  roomInfo: styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
  `,
  roomName: styled.p`
    color: var(--Text-grayDark, #2c2c2e);
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 400;
    line-height: 1.125rem;
  `,
  latestTime: styled.p`
    color: var(--Text-gray, #666668);
    font-family: 'Noto Sans KR';
    font-size: 0.75rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
  `,
  menu: styled.div`
    width: 1rem;
    height: 1rem;
    flex-shrink: 0;
    background: url('/kebab-horizontal.svg') no-repeat;
    cursor: pointer;
  `,
  messageContainer: styled.div`
    display: flex;
    flex: 1;
    flex-direction: column;
    overflow-y: auto;
    width: 100%;
    height: 100%;
    box-shadow: 0px -1px 0px 0px #e5e5ea inset;
    position: relative;

    &::-webkit-scrollbar {
      width: 0.5rem;
    }
    &::-webkit-scrollbar-thumb {
      background-color: #ced3da;
      border-radius: 4px;
    }
  `,
  senderFrame: styled.div`
    display: flex;
    justify-content: flex-end;
    padding-right: 0.8rem;
    margin: 0.8rem 0;
  `,
  receiverFrame: styled.div`
    display: flex;
    justify-content: flex-start;
    padding-left: 0.8rem;
    margin: 0.8rem 0;
  `,
  messageInput: styled.div`
    width: 100%;
    height: 3rem;
    display: flex;
  `,
  messageInputField: styled.input`
    flex: 1;
    font-size: 1.25rem;
    padding: 0.8rem;
    border: none;
    color: var(--Text-grayDark, #2c2c2e);
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 400;
    line-height: 1.125rem;
    /* box-shadow: 0px -1px 0px 0px #e5e5ea inset; */

    &:focus {
      outline: none;
    }
  `,
  backButton: styled.img`
    width: 1rem;
    height: 1rem;
    cursor: pointer;
  `,
};

interface Content {
  roomId: number;
  message: string;
  sender: string;
  nickname: string;
}

function calTimeDiff(time: string, type: string) {
  const lastTime = new Date(time);
  if (type === 'server') lastTime.setHours(lastTime.getHours() + 9);
  const currentTime = new Date();
  const timeDiff = Math.floor(
    (currentTime.getTime() - lastTime.getTime()) / (1000 * 60),
  );

  if (timeDiff < 60) return `${timeDiff}분 전`;

  if (timeDiff < 60 * 24) return `${Math.floor(timeDiff / 60)}시간 전`;

  return `${Math.floor(timeDiff / (60 * 24))}일 전`;
}

function useInterval(callback: () => void, delay: number) {
  const savedCallback = useRef(callback);

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay != null) {
      const id = setInterval(() => {
        savedCallback.current();
      }, delay);
      return () => {
        clearInterval(id);
      };
    }
    return undefined;
  }, [delay]);
}

export function MobileChattingRoom({
  userId,
  userName,
  roomId,
  roomName,
  lastTime,
  onRoomClick,
}: {
  userId: string | undefined;
  userName: string;
  roomId: number;
  roomName: string;
  lastTime: string;
  onRoomClick: React.Dispatch<React.SetStateAction<boolean>>;
}) {
  const [messages, setMessages] = useState<Content[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [stompClient, setStompClient] = useState<Client | null>(null);
  const [isMenuClick, setIsMenuClick] = useState<boolean>(false);
  const [isBackClick, setIsBackClick] = useState<boolean>(false);

  const auth = useAuthValue();
  const user = userId;
  const [roomData, setRoomData] = useState<
    | [
        {
          messageId: string;
          sender: string;
          message: string;
          createdAt: string;
          nickname: string;
        },
      ]
    | undefined
  >();

  const chattingRoom = useEnterChatRoom(roomId, 0, 10);
  useEffect(() => {
    if (chattingRoom != null) {
      setRoomData(chattingRoom.data?.data);
    }
  }, [chattingRoom]);

  const [time, setTime] = useState<string>(lastTime);
  const [type, setType] = useState<string>('server');

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
        stomp.activate();

        stomp.onConnect = () => {
          console.log('WebSocket 연결이 열렸습니다.');
          stomp.subscribe(`/room/${roomId}`, frame => {
            try {
              setTime(new Date().toISOString());
              setType('client');
              const parsedMessage = JSON.parse(frame.body);
              setMessages(prevMessages => [...prevMessages, parsedMessage]);
            } catch (error) {
              console.error('오류가 발생했습니다:', error);
            }
          });
        };
      } catch (error) {
        console.error('채팅 룸 생성 중 오류가 발생했습니다:', error);
      }
    };

    if (auth?.accessToken != null) {
      void initializeChat();
    }

    return () => {
      if (stompClient != null && stompClient.connected) {
        void stompClient.deactivate();
      }
    };
  }, [auth?.accessToken]);

  const sendMessage = () => {
    if (stompClient !== null && stompClient.connected) {
      const destination = `/send/${roomId}`;

      stompClient.publish({
        destination,
        body: JSON.stringify({
          roomId,
          sender: user,
          message: inputMessage,
          nickname: userName,
        }),
      });
    }

    setInputMessage('');
  };

  const [timeString, setTimeString] = useState(calTimeDiff(lastTime, 'server'));

  useEffect(() => {
    setTimeString(calTimeDiff(time, type));
  }, [time, type]);

  useInterval(() => {
    setTimeString(calTimeDiff(time, type));
  }, 60000);

  const handleMenuClick = () => {
    setIsMenuClick(prev => !prev);
  };

  const { mutate: exit } = useExitChatRoom(roomId);

  const handleBackClick = () => {
    exit();
    setIsBackClick(prev => !prev);
    onRoomClick(isBackClick);
  };

  function handleKeyUp(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.keyCode === 13) {
      sendMessage();
    }
  }

  const messageContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messageContainerRef.current != null) {
      messageContainerRef.current.scrollTop =
        messageContainerRef.current.scrollHeight;
    }
  }, [roomData]);

  useEffect(() => {
    if (messageContainerRef.current != null) {
      messageContainerRef.current.scrollTop =
        messageContainerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <styles.container>
      <styles.header>
        <styles.backButton
          onClick={handleBackClick}
          src="/backward-arrow.png"
        />
        <styles.roomInfo>
          <styles.roomName>{roomName}</styles.roomName>
          <styles.latestTime>{timeString}</styles.latestTime>
        </styles.roomInfo>
        <styles.menu onClick={handleMenuClick} />
        {isMenuClick && (
          <ChatMenu roomId={roomId} onMenuClicked={setIsMenuClick} />
        )}
      </styles.header>
      <styles.messageContainer ref={messageContainerRef}>
        {roomData
          ?.slice()
          .reverse()
          ?.map((message, index) => (
            <div key={index}>
              {message.sender === userId ? (
                <styles.senderFrame>
                  <SenderMessage
                    message={message.message}
                    time={message.createdAt}
                    type="server"
                  />
                </styles.senderFrame>
              ) : (
                <styles.receiverFrame>
                  <ReceiverMessage
                    message={message.message}
                    receiver={message.nickname}
                    time={message.createdAt}
                    type="server"
                  />
                </styles.receiverFrame>
              )}
            </div>
          ))}
        {messages.map((message, index) => (
          <div key={index}>
            {message.sender === userId ? (
              <styles.senderFrame>
                <SenderMessage
                  message={message.message}
                  time={new Date().toISOString()}
                  type="client"
                />
              </styles.senderFrame>
            ) : (
              <styles.receiverFrame>
                <ReceiverMessage
                  message={message.message}
                  receiver={message.nickname}
                  time={new Date().toISOString()}
                  type="client"
                />
              </styles.receiverFrame>
            )}
          </div>
        ))}
      </styles.messageContainer>
      <styles.messageInput>
        <styles.messageInputField
          type="text"
          value={inputMessage}
          onChange={e => {
            setInputMessage(e.target.value);
          }}
          onKeyUp={handleKeyUp}
        />
      </styles.messageInput>
    </styles.container>
  );
}
