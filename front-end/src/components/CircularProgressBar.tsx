'use client';

import { type HTMLAttributes } from 'react';
import styled from 'styled-components';

const styles = {
  container: styled.div`
    .circle-background {
      fill: none;
      stroke: #e1563720;
    }

    .circle-progress {
      fill: none;
      stroke: #e15637;
      stroke-linecap: round;
    }
  `,
};

export function CircularProgressBar({
  percentage,
  strokeWidth,
  diameter,
  className,
}: {
  percentage: number;
  strokeWidth: number;
  diameter: number;
} & HTMLAttributes<HTMLDivElement>) {
  const radius = (diameter - strokeWidth) / 2;
  const cx = diameter / 2;
  const cy = diameter / 2;
  const circumference = 2 * Math.PI * radius;
  const dashArray = `${circumference} ${circumference}`;
  const dashOffset = circumference - (percentage / 100) * circumference;

  return (
    <styles.container className={className}>
      <svg width={diameter} height={diameter}>
        <circle
          className="circle-background"
          cx={cx}
          cy={cy}
          r={radius}
          strokeWidth={strokeWidth}
        />
        <circle
          className="circle-progress"
          cx={cx}
          cy={cy}
          r={radius}
          strokeWidth={strokeWidth}
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{
            strokeDasharray: dashArray,
            strokeDashoffset: dashOffset,
          }}
        />
      </svg>
    </styles.container>
  );
}
