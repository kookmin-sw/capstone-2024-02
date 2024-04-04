'use client';

import styled from 'styled-components';

const styles = {
  container: styled.div`
    width: 100%;
  `,
  section1: styled.div`
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    padding: 0rem 13.75rem;
  `,
  img: styled.img`
    width: 29.1875rem;
    height: 39.9375rem;
    flex-shrink: 0;
    object-fit: contain;

    margin-bottom: -6rem;
  `,
  description: styled.div`
    display: flex;
    flex-direction: column;
    align-items: end;
    gap: 1.75rem;
    width: 27.8125rem;
    margin-top: 6.25rem;
    flex-shrink: 0;

    h1 {
      color: #000;
      text-align: right;
      font-family: 'Noto Sans KR';
      font-size: 2rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
      margin-bottom: 2rem;
    }
    p {
      color: #000;
      text-align: right;
      font-family: 'Noto Sans KR';
      font-size: 1.5rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;
      margin-bottom: 6.5rem;
    }
  `,
  loginButtons: styled.div`
    display: flex;
    flex-direction: column;
    gap: 0.6875rem;

    button {
      all: unset;
      cursor: pointer;
      display: flex;
      justify-content: center;
    }

    img {
      width: 16.5625rem;
      height: 2.6875rem;
      flex-shrink: 0;
      border-radius: 8px;
      object-fit: cover;
    }
  `,
  section2: styled.div`
    position: relative;
    width: 100dvw;
    height: 31.25rem;
    flex-shrink: 0;
    background: #f7f6f9;
  `,
  section3: styled.div`
    display: flex;
    justify-content: end;

    height: 31.25rem;
    padding-right: 21.25rem;
  `,
  imageBox: styled.div`
    display: flex;
    flex-direction: row;
    gap: 1rem;
    width: 17rem;
    height: 41rem;
    flex-shrink: 0;
  `,
  boxColumn: styled.div<{ $margin: number }>`
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: ${({ $margin }) => `${$margin / 16}rem`};

    position: relative;
    top: -5rem;
  `,
  box: styled.div`
    display: flex;
    justify-content: center;
    align-items: center;

    width: 5rem;
    height: 5rem;
    flex-shrink: 0;

    border-radius: 16px;
    background: rgba(247, 246, 249, 0.5);
  `,
  section4: styled.div`
    display: flex;
    position: relative;
    width: 100dvw;
    height: 17.9375rem;
    padding: 10.0625rem 38.71875rem 5.0625rem 39.90625rem;
    justify-content: center;
    align-items: center;
    flex-shrink: 0;
    background: #f7f6f9;
  `,
  findMateButton: styled.button`
    all: unset;
    flex-shrink: 0;
    padding: 0.5rem 1.5rem;
    flex-shrink: 0;

    border-radius: 8px;
    background: #e15637;

    color: #fff;
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
    cursor: pointer;
  `,
};

const images = [
  [
    '',
    '',
    '/icon-popular.svg',
    '/icon-hanok.svg',
    '/icon-building.svg',
    '',
    '',
  ],
  [
    '',
    '',
    '/icon-building.svg',
    '/icon-countryside.svg',
    '/icon-camping.svg',
    '',
    '',
  ],
  ['', '', '/icon-house.svg', '/icon-camping.svg', 'icon-hanok.svg', '', ''],
];

export function LandingPage() {
  return (
    <styles.container>
      <styles.section1>
        <styles.img
          alt="Brazuca Date Night"
          src="https://s3-alpha-sig.figma.com/img/cee1/f195/09c0e7edb4b99b5d73a56afd3391e70c?Expires=1713139200&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=GD32LLH2Mn1hwXGowQ4~ZbzqXVnlrJ~PA42mtUf~8IhrvG6frWHG9Z0EAqzF40h9RUUJ8XYyXpcE8L3~wh2falwIiWzXA6R~3irEUOv6YRN1pCVpj43Aftw5-vr3yBPv2Kxyx7NThdtmOpFn5F~PMvarqDzx7Oig4W0pCMgxaWg6RTCG226xbLvIPyS8x5vnn2KePbi84zSGZfS2kiWoYvD7QEaLcIi29kbcb6t9GEuSOB-rV7nJ2y0uJiGafxRelwthA~w4sOPYRQ~I5xmA1fUbjgjnUiByb9iAU9GPlH9TjHh700aTFdpxy-2j2KpbyB7Tz51f0CoqXiX0TH5SQg__"
        />
        <styles.description>
          <h1>공동주거생활의 A to Z</h1>
          <p>
            마루에서 여러분만을 위한
            <br /> 메이트를 만나보세요
          </p>
          <styles.loginButtons>
            <a
              href={`${process.env.NEXT_PUBLIC_API_URL}/oauth2/authorize/kakao?redirect_uri=${process.env.NEXT_PUBLIC_CLIENT_URL}/login`}
            >
              <img
                alt="kakao"
                src="https://s3-alpha-sig.figma.com/img/3d3c/d1f5/021fb8c1c71bbfcfdbd58571b78fafa3?Expires=1713139200&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=FTAwnytd3IJSGgWXu7yJMR2zsfo4PnCcF9Gdfh7vqYyiCH8X6r74r6DujoZNISOFwozcGVt6ir0AcO8v5OB80YCmIsbpDm6QuSe~gOW9e6CMtDODh4ih7DO3DRu5Z9Fl4kXjIO4De5eoH1rh~S6Aeycar4eJOGCSqgTowxQjbx7vU6zmD0-XwjCa7p4p7BpcPcL2duWbrekHcJ8vLmSUDEjLwG-445rH9k9U-BYMe1KTlzcyhg0HLnitxX~XfW9HSN9VADuidDyp4mwBYPeWG87DRXixuWUSGIdipIYBg~NTFemJIVu8U2UP~TI6NAY9U-aDoP7mFy53RCG2UTdMPg__"
              />
            </a>
            <a
              href={`${process.env.NEXT_PUBLIC_API_URL}/oauth2/authorize/naver?redirect_uri=${process.env.NEXT_PUBLIC_CLIENT_URL}/login`}
            >
              <img
                alt="naver"
                src="https://s3-alpha-sig.figma.com/img/bc24/1c57/afda23776f60b331a41aa67a9eaaf66c?Expires=1713139200&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=LuD7rvKOi7gYpAsadxPyl7lq4TXQsumgdHexE9YdScyyDAOQ6m03yzmkwFw1lOI9CTF87sIjT~aywIl2fZKkjcK5Udx4UGgacCsebVPWBfZDEibev75yhf0f-ASFamR6Tz6MDKxxzBhWyD1aUdrdMeanKXdK-mKsyY1vMY1gLbPziG~PHXfyUO-3u~FcbnH4FPIfm47jA1qPooGBgMWCg8HdAVRaBexlG3CtPioxKmwG7fm-Xw6PMy~Nw5USxzVipHT-rgmwNGbeWNLhZo4Unrnmyn38vF3rv0bnJVVeDvJT0XWs2oDmch5Bar1GAhpxH3Ojr4HjKqkHSXTNsEXrEw__"
              />
            </a>
          </styles.loginButtons>
        </styles.description>
      </styles.section1>
      <styles.section2 />
      <styles.section3>
        <styles.imageBox>
          {images.map((column, i) => (
            <styles.boxColumn key={JSON.stringify(column)} $margin={16 * i}>
              {column.map(url =>
                url !== '' ? (
                  <styles.box key={url}>
                    <img alt={url} src={url} />
                  </styles.box>
                ) : (
                  <styles.box key={crypto.randomUUID()} />
                ),
              )}
            </styles.boxColumn>
          ))}
        </styles.imageBox>
      </styles.section3>
      <styles.section4>
        <styles.findMateButton>메이트 찾아보기</styles.findMateButton>
      </styles.section4>
    </styles.container>
  );
}
