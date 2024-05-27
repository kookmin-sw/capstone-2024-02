'use client';

import { useEffect, useState } from 'react';
import styled from 'styled-components';

const styles = {
  container: styled.div`
    display: flex;

    width: 13.625rem;
    height: 2.5rem;
    flex-shrink: 0;

    border-radius: 20px;
    background: var(--Gray-6, #efefef);
    padding: 0.625rem 1rem;
    gap: 0.5rem;
  `,
  input: styled.input`
    all: unset;

    font-size: 0.8rem;
  `,
};

export function SearchBox({
  onContentChange,
  onEnter,
}: {
  onContentChange: React.Dispatch<React.SetStateAction<string>>;
  onEnter: React.Dispatch<React.SetStateAction<boolean>>;
}) {
  const [content, setContent] = useState('');

  useEffect(() => {
    onContentChange(content);
  }, [content]);

  function handleKeyUp(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.keyCode === 13) {
      onEnter(true);
    }
  }

  return (
    <styles.container>
      <img alt="" src="/icon-search.svg" />
      <styles.input
        value={content}
        onChange={e => {
          setContent(e.target.value);
        }}
        onKeyUp={handleKeyUp}
      />
    </styles.container>
  );
}
