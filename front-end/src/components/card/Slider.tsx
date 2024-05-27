'use client';

import React, { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';

const styles = {
  container: styled.div`
    display: inline-flex;
    width: 50%;
    align-items: center;
  `,
  sliderContainer: styled.div`
    width: 22rem;
    height: 1.875rem;
    position: relative;

    @media (max-width: 768px) {
      width: 100%;
      height: 1.5rem;
    }
  `,
  sliderTrack: styled.div`
    width: 100%;
    height: 0.3125rem;
    border-radius: 20px;
    background: #d9d9d9;
    position: absolute;
    top: calc(50% - 2px);
  `,
  sliderFillTrack: styled.div<FillProps>`
    width: ${props => props.$fill};
    height: 0.3125rem;
    border-radius: 2px;
    background: var(--Main-1, #e15637);
    position: absolute;
    top: calc(50% - 2px);
    left: ${props => props.$left};
    right: ${props => props.$right};
  `,
  slider: styled.input`
    position: absolute;
    width: 100%;
    height: 0.3125rem;
    border-radius: 1.25rem;
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    background: transparent;
    top: calc(50% - 2px);

    &:focus {
      outline: none;
    }

    &::-webkit-slider-thumb {
      -webkit-appearance: none;
      pointer-events: all;
      width: 1.875rem;
      height: 1.875rem;
      background-color: #fff;
      border-radius: 50%;
      box-shadow: 0 0 0 1px #c6c6c6;
      cursor: pointer;
      position: relative;
      z-index: 1;
      box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.2);

      @media (max-width: 768px) {
        width: 1.2rem;
        height: 1.2rem;
      }
    }
  `,
};

interface SliderProps {
  min: number;
  max: number;
  step: number;
  initialMin: number;
  initialMax: number;
  onChange: (min: number, max: number) => void;
}

interface FillProps {
  $fill: string;
  $left: string;
  $right: string;
}

interface SliderProps {
  min: number;
  max: number;
  step: number;
  initialMin: number;
  initialMax: number;
  onChange: (minValue: number, maxValue: number) => void;
}

export function Slider({
  min,
  max,
  step,
  initialMin,
  initialMax,
  onChange,
}: SliderProps) {
  const inputMinRef = useRef<HTMLInputElement>(null);
  const inputMaxRef = useRef<HTMLInputElement>(null);
  const [minValue, setMinValue] = useState(initialMin);
  const [maxValue, setMaxValue] = useState(initialMax);

  useEffect(() => {
    setMinValue(initialMin);
    setMaxValue(initialMax);
  }, [initialMin, initialMax]);

  const handleMinChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newLow = Number(e.target.value);
    const currentMax =
      inputMaxRef.current !== null
        ? Number(inputMaxRef.current.value)
        : initialMin;
    if (newLow > currentMax) {
      setMaxValue(newLow);
      inputMaxRef.current !== null &&
        (inputMaxRef.current.value = newLow.toString());
    }
    setMinValue(newLow);
  };

  const handleMaxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newHigh = Number(e.target.value);
    const currentMin =
      inputMinRef.current !== null
        ? Number(inputMinRef.current.value)
        : initialMax;
    if (newHigh < currentMin) {
      setMinValue(newHigh);
      inputMinRef.current !== null &&
        (inputMinRef.current.value = newHigh.toString());
    }
    setMaxValue(newHigh);
  };

  useEffect(() => {
    onChange(minValue, maxValue);
  }, [onChange, minValue, maxValue]);

  return (
    <styles.container>
      <styles.sliderContainer>
        <styles.sliderTrack />
        <styles.sliderFillTrack
          $fill={`${((maxValue - minValue) / (max - min)) * 100}%`}
          $left={`${(minValue / max) * 100}%`}
          $right={`${100 - (maxValue / max) * 100}%`}
        />
        <styles.slider
          type="range"
          ref={inputMinRef}
          min={min}
          max={max}
          step={step}
          value={minValue ?? min}
          onChange={handleMinChange}
        />
        <styles.slider
          type="range"
          ref={inputMaxRef}
          min={min}
          max={max}
          step={step}
          value={maxValue ?? max}
          onChange={handleMaxChange}
        />
      </styles.sliderContainer>
    </styles.container>
  );
}
