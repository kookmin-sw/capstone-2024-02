'use client';

import { useState } from 'react';
import styled from 'styled-components';

import { CircularButton } from '@/components';

const styles = {
  container: styled.div`
    position: relative;
    display: flex;
    align-items: center;
    width: 100%;
    height: 100%;
  `,
  buttonContainer: styled.div`
    position: absolute;

    width: 100%;
    display: flex;
    justify-content: space-between;

    padding: 0.69rem;
  `,
  imageContainer: styled.div`
    width: 100%;

    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-auto-rows: 20.625rem;
    gap: 0.81rem;

    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .left {
      border-radius: 16px 0 0 16px;
    }

    .right {
      border-radius: 0 16px 16px 0;
    }
  `,
  none: styled.div`
    width: 100%;
    height: 100%;
    background: var(--Gray-6, #efefef);
  `,
};

interface Props {
  images: string[];
}

const sliceImages = (images: string[]) => {
  const result: string[][] = [];
  for (let i = 0; i < images.length; i += 2) {
    if (i + 1 < images.length) {
      result.push([images[i], images[i + 1]]);
    } else {
      result.push([images[i], 'none']);
    }
  }
  return result;
};

export function ImageGrid({ images: imagesParam }: Props) {
  const [images] = useState(sliceImages(imagesParam));
  const [pageIndex, setPageIndex] = useState(0);

  const handlePageChange = (diff: number) => {
    if (pageIndex + diff < 0 || pageIndex + diff >= images.length) {
      return;
    }

    setPageIndex(pageIndex + diff);
  };

  return (
    <styles.container>
      <styles.buttonContainer>
        <CircularButton
          disabled={pageIndex - 1 < 0}
          direction="left"
          onClick={() => {
            handlePageChange(-1);
          }}
        />
        <CircularButton
          disabled={pageIndex + 1 >= images.length}
          direction="right"
          onClick={() => {
            handlePageChange(1);
          }}
        />
      </styles.buttonContainer>
      <styles.imageContainer>
        {images[pageIndex].map((image, index) =>
          image !== 'none' ? (
            <img
              key={image}
              alt={image}
              src={image}
              className={index % 2 === 0 ? 'left' : 'right'}
            />
          ) : (
            <styles.none key="none" className="right" />
          ),
        )}
      </styles.imageContainer>
    </styles.container>
  );
}
