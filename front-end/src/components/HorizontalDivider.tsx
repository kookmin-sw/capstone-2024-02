'use client';

import styled from 'styled-components';

const styles = {
  container: styled.div`
    width: 100%;
    height: 1px;
    border: 1px solid #d3d0d7;
    border-right-width: 0px;
    border-bottom-width: 0px;
    border-left-width: 0px;
  `,
};

export function HorizontalDivider() {
  return <styles.container />;
}
