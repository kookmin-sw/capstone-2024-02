'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';

interface StyleProps {
  $width: number;
  $margin: number;
}

const styles = {
  container: styled.div<StyleProps>`
    position: relative;
    width: 100%;

    div {
      position: absolute;
      left: 0;
      height: 2px;
      background-color: gray;
      width: 100%;
    }

    div[class~='range'] {
      position: absolute;
      height: 2px;
      background-color: #e15637;
      width: ${({ $width }) => `${$width}px`};
      left: ${({ $margin }) => `${$margin}px`};
    }

    input[type='range']::-webkit-slider-thumb {
      -webkit-appearance: none;
      pointer-events: all;
      width: 24px;
      height: 24px;
      background-color: #fff;
      border-radius: 50%;
      box-shadow: 0 0 0 1px #c6c6c6;
      cursor: pointer;
    }

    input[type='range']::-moz-range-thumb {
      -webkit-appearance: none;
      pointer-events: all;
      width: 24px;
      height: 24px;
      background-color: #fff;
      border-radius: 50%;
      box-shadow: 0 0 0 1px #c6c6c6;
      cursor: pointer;
    }

    input[type='range']::-webkit-slider-thumb:hover {
      background: #f7f7f7;
    }

    input[type='range']::-webkit-slider-thumb:active {
      box-shadow:
        inset 0 0 3px #e15637,
        0 0 9px #e15637;
      -webkit-box-shadow:
        inset 0 0 3px #e15637,
        0 0 9px #e15637;
    }

    input[type='range'] {
      -webkit-appearance: none;
      appearance: none;
      height: 2px;
      width: 100%;
      position: absolute;
      background-color: transparent;
      pointer-events: none;
    }
  `,
};

interface Props {
  min: number;
  max: number;
  onChange: ({ low, high }: { low: number; high: number }) => void;
}

export function RangeSlider({ min, max, onChange }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  const [low, setLow] = useState(min);

  const [high, setHigh] = useState(max);

  const { width, margin }: { width: number; margin: number } = useMemo(() => {
    if (containerRef?.current === null) {
      return { width: 100, margin: 0 };
    }

    const containerWidth = containerRef.current.clientWidth;
    const highPercentage = high / 100;
    const lowPercentage = low / 100;

    return {
      width: containerWidth * (highPercentage - lowPercentage),
      margin: containerWidth * lowPercentage,
    };
  }, [containerRef, low, high]);

  useEffect(() => {
    setLow((min + max) * 0.25);
    setHigh((min + max) * 0.75);
  }, []);

  useEffect(() => {
    onChange({ low, high });
  }, [onChange, low, high]);

  return (
    <styles.container ref={containerRef} $width={width} $margin={margin}>
      <div />
      <div className="range" />
      <input
        value={low}
        onChange={e => {
          const newLow = +e.currentTarget.value;
          if (newLow > high) setHigh(newLow);
          setLow(newLow);
        }}
        className="low"
        type="range"
      />
      <input
        value={high}
        onChange={e => {
          const newHigh = +e.currentTarget.value;
          if (newHigh < low) setLow(newHigh);
          setHigh(newHigh);
        }}
        className="high"
        type="range"
      />
    </styles.container>
  );
}
