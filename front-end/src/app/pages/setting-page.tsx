'use client';

import styled from 'styled-components';

import { VitalSection, OptionSection } from '@/components';

const styles = {
  pageContainer: styled.div`
    display: flex;
    flex-direction: column;
    height: 85rem;
  `,

  cardName: styled.p`
    margin: 4.5625rem 0;
    color: #000;

    font-family: 'Noto Sans KR';
    font-size: 2rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,

  cardContainer: styled.div`
    display: flex;
    gap: 0.88rem;
  `,
  miniCard: styled.div`
    width: 23.0625rem;
    height: 17.5rem;
    flex-shrink: 0;
    border-radius: 1.875rem;
    background: #f7f6f9;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    padding: 1.62rem 1.44rem;
    display: flex;
    flex-direction: column;
  `,
  miniCardName: styled.p`
    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 1.5rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    margin-bottom: 1.69rem;
  `,
  miniCardKeywordsContainer: styled.div`
    width: 17.5625rem;
    height: 5.6875rem;
    position: relative;
  `,
  miniCardKeyword: styled.div`
    display: inline-flex;
    padding: 0.5rem 1.5rem;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    border-radius: 1.625rem;
    border: 2px solid var(--Main-1, #e15637);
    background: #fff;

    color: var(--Main-1, #e15637);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    position: absolute;
  `,
  checkContainer: styled.div`
    width: 51.0625rem;
    height: 95.8125rem;
    flex-shrink: 0;
    border-radius: 1.875rem;
    background: var(--background, #f7f6f9);
    padding: 3.56rem 0 0 1.56rem;
    margin-bottom: 7.5rem;
  `,

  lineContainer: styled.div`
    margin: 2.69rem 0;
    padding: 0 4.37rem 0 1.38rem;
  `,
  horizontalLine: styled.hr`
    width: 100%;
    height: 0rem;
    flex-shrink: 0;
    stroke-width: 1px;
    stroke: #d3d0d7;
  `,
};

interface SettingPageProps {
  type: string;
  name: string;
}

export function SettingPage({ type, name }: SettingPageProps) {
  return (
    <styles.pageContainer>
      <styles.cardName>
        {type} 카드 &gt; {name}
      </styles.cardName>
      <styles.cardContainer>
        <styles.miniCard>
          <styles.miniCardName>{type}카드</styles.miniCardName>
          <styles.miniCardKeywordsContainer>
            <styles.miniCardKeyword>여성</styles.miniCardKeyword>
            <styles.miniCardKeyword style={{ right: '0' }}>
              비흡연
            </styles.miniCardKeyword>
            <styles.miniCardKeyword style={{ bottom: '0' }}>
              아침형
            </styles.miniCardKeyword>
          </styles.miniCardKeywordsContainer>
        </styles.miniCard>
        <styles.checkContainer>
          <VitalSection />
          <styles.lineContainer>
            <styles.horizontalLine />
          </styles.lineContainer>
          <OptionSection />
        </styles.checkContainer>
      </styles.cardContainer>
    </styles.pageContainer>
  );
}
