'use client';

import styled from 'styled-components';

import { getLocalTime } from '@/shared';

const styles = {
  container: styled.div`
    display: inline-flex;
    align-items: center;
  `,
  right: styled.div`
    display: flex;
    padding: 0.25rem 0rem 0.25rem 0.5rem;
    align-items: center;
    border-radius: 6px 0 0 6px;
    background: var(--Gray-5, #828282);
  `,
  messageFrame: styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-end;
  `,
  messageBody: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    max-width: 20rem;
    gap: 0.25rem;
  `,
  message: styled.p`
    width: 100%;
    max-width: 20rem;
    color: var(--White, #fff);
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 400;
    line-height: 1.25rem;
    word-break: break-word;
    white-space: pre-wrap;
  `,
  messageInfo: styled.div`
    display: flex;
    width: 3.5rem;
    align-items: flex-end;
    gap: 0.25rem;
    align-self: stretch;
  `,
  time: styled.span`
    color: var(--White, #fff);
    font-family: 'Noto Sans KR';
    font-size: 0.75rem;
    font-style: normal;
    font-weight: 300;
    line-height: normal;
  `,
  left: styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    align-self: stretch;
  `,
  leftTop: styled.div`
    width: 0.9375rem;
    height: 0.75rem;
    background: url('/Bubble tip.svg') no-repeat;
  `,
  leftMiddle: styled.div`
    width: 0.375rem;
    flex: 1 0 0;
    background: var(--Gray-5, #828282);
  `,
  leftBottom: styled.div`
    width: 0.375rem;
    height: 0.4375rem;
    fill: var(--Gray-5, #828282);
    background: url('/bottom-curve-vector.svg') no-repeat;
  `,
};

export function SenderMessage({
  message,
  time,
  type,
}: {
  message: string;
  time: string;
  type: string;
}) {
  return (
    <styles.container>
      <styles.right>
        <styles.messageFrame>
          <styles.messageBody>
            <styles.message>{message} </styles.message>
            <styles.time>{getLocalTime(time, type)}</styles.time>
          </styles.messageBody>
        </styles.messageFrame>
      </styles.right>
      <styles.left>
        <styles.leftTop />
        <styles.leftMiddle />
        <styles.leftBottom />
      </styles.left>
    </styles.container>
  );
}
