'use client';

import { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';

import { fromAddrToCoord, type NaverAddress } from '@/features/geocoding';

const styles = {
  container: styled.div`
    position: fixed;
    width: 50dvw;
    height: 50dvh;
    background-color: white;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    border-radius: 16px;
    padding: 2rem;

    left: 50%;
    top: 50%;
    translate: -50% -50%;

    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;

    display: flex;
    flex-direction: column;
    gap: 1rem;

    @media (max-width: 768px) {
      padding: 1.2rem;
    }
  `,
  title: styled.h1`
    font-size: 1.25rem;
    margin-bottom: 1rem;

    @media (max-width: 768px) {
      font-size: 0.875rem;
    }
  `,
  locationInput: styled.input<{ $empty: boolean }>`
    width: 100%;
    height: fit-content;
    padding: 1rem;

    border: none;
    border-radius: 8px;
    background: var(--Gray-6, #efefef);

    color: ${({ $empty }) => ($empty ? '#9a95a3' : '#000')};

    font-size: 1.1rem;
    &:focus {
      outline: none;
    }

    @media (max-width: 768px) {
      padding: 0.5rem;
      font-size: 0.75rem;
    }
  `,
  addressSearchResult: styled.ul`
    display: flex;
    flex-direction: column;
    flex: 1 0 0;
    background: var(--Gray-6, #efefef);
    border-radius: 8px;
    padding: 1rem;
  `,
  addressItem: styled.li`
    width: 100%;
    list-style-type: none;

    padding-bottom: 0.5rem;
    border-bottom: 0.5px solid #19191980;

    cursor: pointer;

    @media (max-width: 768px) {
      padding: 0.5rem;
      font-size: 0.75rem;
    }
  `,
};

export function LocationSearchBox({
  onSelect,
  setHidden,
}: {
  onSelect: (address: NaverAddress) => void;
  setHidden: () => void;
}) {
  const [searchText, setSearchText] = useState<string>('');
  const [addresses, setAddresses] = useState<NaverAddress[]>([]);

  const containerRef = useRef<HTMLDivElement>(null);
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
      <form
        onSubmit={event => {
          event.preventDefault();
          fromAddrToCoord({ query: searchText })
            .then(response => {
              setAddresses(response);
            })
            .catch((error: Error) => {
              console.log(error);
            });
        }}
      >
        <styles.title>주소 검색</styles.title>
        <styles.locationInput
          placeholder="주소 입력"
          $empty={searchText.length === 0}
          value={searchText}
          onChange={event => {
            setSearchText(event.target.value);
          }}
        />
      </form>
      <styles.addressSearchResult>
        {addresses.map(address => (
          <styles.addressItem
            onClick={() => {
              onSelect(address);
            }}
            key={address.roadAddress}
          >
            {address.roadAddress}
          </styles.addressItem>
        ))}
      </styles.addressSearchResult>
    </styles.container>
  );
}
