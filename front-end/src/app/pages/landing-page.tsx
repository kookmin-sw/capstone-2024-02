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
          src="https://s3-alpha-sig.figma.com/img/cee1/f195/09c0e7edb4b99b5d73a56afd3391e70c?Expires=1714348800&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=mGbLX64XEb4~I8MnA4qswCUMllOL~P7U2jkKUuzLYa6x3Fr2kB6NFyORGvAhWNjkQ1PGIkTB9bul-F4o2hdS-jE5~1y0Ss1sLqK-j8bFMsFvJSm7qqTShNsBKHYXnbXA0d8tl74LDrhwNADsVQbR2p7xG~syVQ3l-NSmzaGaeEF3g0tnUIB3dybwZY1llwLcazj7yBXhVY-NMWg8ekFaFNU7ZRsK7HhK81bQJb1CBNughUXF1O1xDvQZjh3IKx~f2TP5qJAkB5Z1Nw9i5SoZF9YEo-odhDSrvzy7Gbfqw9W0xyXesJVGTRF~AZ3z8D5Zn2E~aZDvg5i3SWJHfFuFXg__"
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
                src="https://s3-alpha-sig.figma.com/img/3d3c/d1f5/021fb8c1c71bbfcfdbd58571b78fafa3?Expires=1714348800&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=Jgm070ukGhSkHRxajO10LZQixoKIVm2S6l7xuP6MY0UJD54Rosq2xvca-UgdiThbu-TLnIu2xuoP0gOXtw-L4PDccdMPBnaURCuygfJSH6ELelnq12~hR9YuwNLifCAl48tAnjwwkCZCPlGlbK1HiUIhAZ5HnF2qXc2PQjq4Zfu0WBZOhttWhRRlZhmHOFFrus--5xYMq4lIw6n0Or8aGJ~Amf9HDlVQmj4ioeSG9HZB3DAEpoIcgqhzzSjqG64Z-U5k~cYZc8V~y5qmm3bVX1n3HCfKi4F~25zmS3A9tNX~y2VcaS-L018gcMtHVke9Qb6C0B9foBMZI6TobWy-Ng__"
              />
            </a>
            <a
              href={`${process.env.NEXT_PUBLIC_API_URL}/oauth2/authorize/naver?redirect_uri=${process.env.NEXT_PUBLIC_CLIENT_URL}/login`}
            >
              <img
                alt="naver"
                src="https://s3-alpha-sig.figma.com/img/bc24/1c57/afda23776f60b331a41aa67a9eaaf66c?Expires=1714348800&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=dk8rQuwlcSV6HVLJZQ3SI4Cf5tGL8HbnfAisVrT5kE5-rqlQ71CKGQtz0xTKX-7bt4OVOOsMaaQX5IsqhHnX2ZEudp4AN2M6lGQf9lE9iLvKT~5uAnYiwMHCXIJ-zuu6qLpDu~rDAPZ92ki8Bc6Dp-OIxkWU75wb54T7erfxNZgATHwEBre-5I7~pPWo-GETubNWL7zF~QFP97Cqf34gORT7GlRg~qPHnBJfeER7AcExfyXG8CXyqjkk5uudH5FD7zU3a9PGunNQDRdComHZRx6MOQBltHa6mPT78MBB8mPyDlJxuRNrpkyFJKDa3I3JWwL-Ns-qJlwBW0WDnPsYQA__"
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
