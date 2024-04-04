'use client';

import styled from 'styled-components';

const styles = {
  container: styled.img<{ $isHost: boolean }>`
    display: flex;
    width: 5.25rem;
    height: 5.25rem;
    justify-content: center;
    align-items: center;
    flex-shrink: 0;

    border-radius: 100px;
    border: 1px solid ${({ $isHost }) => ($isHost ? '#e15637' : 'DCDDEA')};
    background: #c4c4c4;

    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    cursor: pointer;

    object-fit: center;
  `,
};

export function MiniCircularProfileImage({
  isHost,
  url,
  style,
}: {
  isHost: boolean;
  url: string;
} & React.ComponentProps<'div'>) {
  return <styles.container style={style} $isHost={isHost} alt="" src={url} />;
}
