'use client';

import { useEffect, useState } from 'react';
import styled from 'styled-components';

const styles = {
  container: styled.div<{ $showList: boolean }>`
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;

    position: relative;
    width: fit-content;
    height: fit-content;

    display: flex;
    flex-direction: column;
    padding: 0.75rem 1.25rem;

    border-radius: ${({ $showList }) => ($showList ? '12px 12px 0 0' : '12px')};
    border: 2px solid var(--Gray-4, #dfdfdf);
    border-bottom-width: ${({ $showList }) => ($showList ? '0' : '2px')};

    cursor: pointer;
  `,
  content: styled.div`
    display: flex;
    flex-direction: row;
    gap: 0.5rem;

    color: var(--Gray-3, #888);
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,
  list: styled.div`
    position: absolute;
    top: 100%;
    left: -2px;
    padding: 0.75rem 1.25rem;

    display: flex;
    width: calc(100% + 4px);
    flex-direction: column;
    gap: 0.5rem;

    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    border: 2px solid var(--Gray-4, #dfdfdf);
    border-top-width: 0px;
    border-radius: 0 0 12px 12px;
    z-index: 1;
  `,
  item: styled.button`
    all: unset;
  `,
};

interface Props {
  title: string;
  items: string[];
  selected?: string;
  onSelect: (item: string) => void;
}

export function DropDownList({ title, items, selected, onSelect }: Props) {
  const [showList, setShowList] = useState(false);

  useEffect(() => {
    const handler = () => {
      setShowList(false);
    };

    document.addEventListener('click', handler);
    return () => {
      document.removeEventListener('click', handler);
    };
  }, [showList, setShowList]);

  return (
    <styles.container
      $showList={showList}
      onClick={() => {
        setShowList(true);
      }}
    >
      <styles.content>
        {selected ?? title}
        <img alt="drop-down-button-2" src="/icon-drop-down_2.svg" />
      </styles.content>
      {showList && (
        <styles.list>
          {items.map(item => (
            <styles.item
              onClick={() => {
                onSelect(item);
              }}
              key={item}
            >
              {item}
            </styles.item>
          ))}
        </styles.list>
      )}
    </styles.container>
  );
}
