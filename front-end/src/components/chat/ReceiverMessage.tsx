'use client';

import styled from 'styled-components';

import { getLocalTime } from '@/shared';

const styles = {
  container: styled.div`
    display: flex;
    align-items: flex-start;
    gap: 0.25rem;
    flex-shrink: 0;
  `,
  profileSection: styled.div`
    display: flex;
    flex-direction: column;
    width: 3rem;
    height: 3rem;
    justify-content: center;
    align-items: center;
    flex-shrink: 0;
  `,
  profilePic: styled.div`
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    flex-shrink: 0;
    background: url('/__avatar_url.png') center no-repeat;
  `,
  userName: styled.p`
    color: var(--Text-gray, #666668);
    font-family: 'Noto Sans KR';
    font-size: 0.75rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
  `,
  messageContainer: styled.div`
    display: flex;
    align-items: center;
  `,
  left: styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-end;
    align-self: stretch;
  `,
  leftTop: styled.div`
    width: 0.9375rem;
    height: 0.75rem;
    background: url('/Bubble tip R.svg') no-repeat;
  `,
  leftMiddle: styled.div`
    width: 0.375rem;
    flex: 1 0 0;
    background: var(--gray-gray-1, #f2f2f7);
  `,
  leftBottom: styled.div`
    width: 0.375rem;
    height: 0.4375rem;
    background: url('/bottom-curve-vector R.svg') no-repeat;
  `,
  right: styled.div`
    display: flex;
    padding: 0.25rem 0.5rem 0.25rem 0rem;
    align-items: center;
    border-radius: 0px 6px 6px 0px;
    background: var(--gray-gray-1, #f2f2f7);
  `,
  messageFrame: styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-end;
    gap: 0.25rem;
    align-self: stretch;
  `,
  messageBody: styled.div`
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
  `,
  message: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.625rem;
    color: var(--Text-gray, #666668);
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 400;
    line-height: 1.25rem;
  `,
  time: styled.p`
    color: var(--Text-gray, #666668);
    font-family: 'Noto Sans KR';
    font-size: 0.75rem;
    font-style: normal;
    font-weight: 300;
    line-height: normal;
  `,
  messageInfo: styled.div`
    display: flex;
    margin-top: 2rem;
    align-items: flex-end;
    gap: 0.25rem;
    align-self: stretch;
  `,
};

export function ReceiverMessage({
  message,
  receiver,
  time,
  type,
}: {
  message: string;
  receiver: string;
  time: string;
  type: string;
}) {
  return (
    <styles.container>
      <styles.profileSection>
        <styles.profilePic />
        <styles.userName>{receiver}</styles.userName>
      </styles.profileSection>
      <styles.messageContainer>
        <styles.left>
          <styles.leftTop />
          <styles.leftMiddle />
          <styles.leftBottom />
        </styles.left>
        <styles.right>
          <styles.messageFrame>
            <styles.messageBody>
              <styles.message>{message}</styles.message>
              <styles.messageInfo>
                <styles.time>{getLocalTime(time, type)}</styles.time>
              </styles.messageInfo>
            </styles.messageBody>
          </styles.messageFrame>
        </styles.right>
      </styles.messageContainer>
    </styles.container>
  );
}
