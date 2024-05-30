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
    margin-block: 1rem;

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
      appearance: none;
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
        inset 0 0 1px black,
        0 0 1px black;
      -webkit-box-shadow:
        inset 0 0 1px black,
        0 0 1px black;
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
  step: number;
  onChange: ({ low, high }: { low: number; high: number }) => void;
}

export function RangeSlider({ min, max, step, onChange }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  const [state, setState] = useState<{ low: number; high: number }>({
    low: min,
    high: max,
  });

  const { width, margin }: { width: number; margin: number } = useMemo(() => {
    if (containerRef?.current === null) {
      return { width: 100, margin: 0 };
    }

    const containerWidth = containerRef.current.clientWidth;
    const highPercentage = state.high / (min + max - 1);
    const lowPercentage = state.low / (min + max - 1);

    return {
      width: containerWidth * (highPercentage - lowPercentage),
      margin: containerWidth * lowPercentage,
    };
  }, [containerRef, state, max, min]);

  useEffect(() => {
    // width 값을 초기에 계산하기 위해 추가된 effect.
    setState({ low: min, high: max });
  }, [setState, min, max]);

  return (
    <styles.container ref={containerRef} $width={width} $margin={margin}>
      <div />
      <div className="range" />
      <input
        step={step}
        min={min}
        max={max}
        value={state.low}
        onChange={e => {
          const newLow = +e.currentTarget.value;
          const newState = { ...state };

          newState.low = newLow;
          if (newLow > state.high) newState.high = newLow;

          onChange(newState);
          setState(newState);
        }}
        className="low"
        type="range"
      />
      <input
        step={step}
        min={min}
        max={max}
        value={state.high}
        onChange={e => {
          const newHigh = +e.currentTarget.value;
          const newState = { ...state };

          newState.high = newHigh;
          if (newHigh < state.low) newState.low = newHigh;

          onChange(newState);
          setState(newState);
        }}
        className="high"
        type="range"
      />
    </styles.container>
  );
}
