'use client';

import { useEffect, useLayoutEffect, useRef } from 'react';
import styled, { keyframes } from 'styled-components';

import { useToast } from '@/features/toast';

const toastAppeared = keyframes`
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
`;

const toastDisappeared = keyframes`
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
`;

const styles = {
  providerContainer: styled.div`
    position: fixed;
    left: 2rem;
    bottom: 1rem;

    display: flex;
    width: fit-content;
    height: fit-content;
    flex-direction: column;
    gap: 0.5rem;
  `,
  toastContainer: styled.div<{ $isVisible: boolean }>`
    width: fit-content;
    height: fit-content;

    color: white;
    background-color: #e15637;

    padding: 1rem;
    border-radius: 8px;

    animation: ${({ $isVisible }) =>
        $isVisible ? toastAppeared : toastDisappeared}
      200ms ease-in-out;
    z-index: 2147483647;
  `,
  toastMessageContainer: styled.div`
    min-width: 5rem;
    width: fit-content;

    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-weight: bold;
  `,
};

export function ToastProvider() {
  const { toast } = useToast();
  const rectMap = useRef<Map<string, DOMRect>>(new Map()).current;

  useEffect(() => {
    toast.forEach(({ id }) => {
      const element = document.getElementById(id);
      if (element != null)
        rectMap.set(element.id, element.getBoundingClientRect());
    });
  }, []);

  useLayoutEffect(() => {
    toast.forEach(({ id }) => {
      if (rectMap.get(id) == null) {
        const element = document.getElementById(id);
        if (element == null) return;
        rectMap.set(element.id, element.getBoundingClientRect());
      }

      const cacheRect = rectMap.get(id);
      const element = document.getElementById(id);
      if (element == null || cacheRect == null) return;

      const rect = element.getBoundingClientRect();

      const animation = element.animate(
        [
          {
            transform: `translateY(${cacheRect.top - rect.top}px)`,
          },
          {
            transform: `translateY(0px)`,
          },
        ],
        { duration: 200, easing: 'ease-in-out' },
      );

      animation.addEventListener('finish', event => {
        rectMap.set(id, rect);
      });
    });
  }, [toast]);

  return (
    <styles.providerContainer>
      {toast.map(toastMessage => (
        <styles.toastContainer
          id={toastMessage.id}
          key={toastMessage.id}
          $isVisible={toastMessage.isVisible}
        >
          <styles.toastMessageContainer>
            {toastMessage.message}
          </styles.toastMessageContainer>
        </styles.toastContainer>
      ))}
    </styles.providerContainer>
  );
}
