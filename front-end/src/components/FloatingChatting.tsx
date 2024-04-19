'use client';

import Image from 'next/image';
import React, { useState } from 'react';
import styled from 'styled-components';

const styles = {
  chattingButton: styled.div`
    position: fixed;
    bottom: 1.5rem;
    right: 3rem;
    display: inline-flex;
    padding: 1rem;
    align-items: flex-start;
    gap: 0.5rem;
    border-radius: 100px;
    background: #e15637;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    z-index: 100;
    transition: transform 0.3s ease;
    cursor: pointer;

    &:hover {
      transform: scale(1.1);
    }
  `,
  chattingContainer: styled.div`
    position: fixed;
    bottom: 5rem;
    right: 6.5rem;
    display: flex;
    width: 25%;
    height: 70%;
    flex-direction: column;
    align-items: flex-start;
    flex-shrink: 0;
    border-radius: 20px;
    background: #fff;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    z-index: 100;
    overflow: 'auto';
    -webkit-transition: 0.4s;
    transition: 0.4s;
  `,
  chattingHeader: styled.div`
    width: 100%;
    height: 3.25rem;
    flex-shrink: 0;
    border-radius: 20px 20px 0 0;
    background: var(--background, #f7f6f9);
  `,
};

export function FloatingChatting() {
  const [isChatOpen, setIsChatOpen] = useState<boolean>(false);

  const toggleChat = () => {
    setIsChatOpen(prevState => !prevState);
  };
  return (
    <>
      <styles.chattingButton onClick={toggleChat}>
        <Image src="/chatting.svg" width={25} height={25} alt="chatting-icon" />
      </styles.chattingButton>
      {isChatOpen && (
        <styles.chattingContainer>
          <styles.chattingHeader />
        </styles.chattingContainer>
      )}
    </>
  );
}
