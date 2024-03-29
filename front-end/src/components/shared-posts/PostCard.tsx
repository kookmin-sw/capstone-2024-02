'use client';

import styled from 'styled-components';

import { HorizontalDivider } from '..';

const styles = {
  container: styled.div`
    width: 100%;
    height: 15rem;

    display: flex;
    gap: 1.56rem;
  `,
  thumbnail: styled.img`
    width: 16.125rem;
    height: 12.5rem;
    flex-shrink: 0;
    border-radius: 16px;

    object-fit: cover;
  `,
  content: styled.div`
    flex-grow: 1;

    display: flex;
    flex-direction: column;
    gap: 1rem;

    div {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;

      h1 {
        color: #000;
        font-family: 'Noto Sans KR';
        font-size: 1.25rem;
        font-style: normal;
        font-weight: 700;
        line-height: normal;
      }

      h2 {
        color: var(--Gray-5, #828282);
        font-family: 'Noto Sans KR';
        font-size: 1.125rem;
        font-style: normal;
        font-weight: 400;
        line-height: normal;
      }

      p {
        color: var(--Black, #35373a);
        font-family: 'Noto Sans KR';
        font-size: 1rem;
        font-style: normal;
        font-weight: 500;
        line-height: normal;
      }
    }
  `,
  writer: styled.div`
    position: relative;

    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;

    img {
      border-radius: 6.25rem;
      border: 1px solid #dcddea;
      background: #c4c4c4;

      width: 6.25rem;
      height: 6.25rem;
      flex-shrink: 0;
      border-radius: 50%;

      object-fit: cover;
    }

    p {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1.5rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;
    }
  `,
  percentage: styled.div`
    position: absolute;
    left: 75%;
    top: 30%;

    display: inline-flex;
    padding: 0.75rem 0.375rem;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;

    border-radius: 50%;
    background: #fff;
    box-shadow: 0px 4px 4px 0px rgba(0, 0, 0, 0.04);

    p {
      color: #e15637;
      font-family: Pretendard;
      font-size: 1rem;
      font-style: normal;
      font-weight: 600;
      line-height: 1.5rem;
    }
  `,
};

export function PostCard() {
  return (
    <div>
      <styles.container>
        <styles.thumbnail
          alt=""
          src="https://s3-alpha-sig.figma.com/img/2d52/bfda/0a900d41e2a6d77dd731fac06577540c?Expires=1712534400&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=Q9zefEFzqJ-RNfzsNdrJTBDLNSZWOpY-3j8DE04rk-t2MmcZieVHD4MJG5EAEVd1PLLCpN~BKGfMyCHGgjGTr6XcjSifghDXqpxM2fUQSWznNWBdmn4ZrylLoeV17NzynTtmPSND8eo5r86Gv5DAHWNWwmnP6kDtApBlJjoeVSzqTiC7JS4COohy1Wb6Z7Du-ra9apbhzsvoYFytJR68WVqYI0eNAltK3iy23Pn-dysqHoLvty67TI60IYQkcRuU93K6bx4W8tZkxtTg7EW-dZ9uOSLSdwFQ2s14WXXKgQQySfbJH2~g1-usTVjv5u4nVBmV-afBaQbyDbPkkksW8Q__"
        />
        <styles.content>
          <div>
            <h1>혼자 살긴 너무 큰 방 같이 살 룸메이트 구해요!</h1>
            <h2>서울특별시 성북구 정릉로 104</h2>
          </div>
          <div>
            <p>모집 1명 / 총원 2명</p>
            <p>원룸 · 방1</p>
            <p>500 / 50 / 5</p>
          </div>
        </styles.content>
        <styles.writer>
          <img
            alt=""
            src="https://s3-alpha-sig.figma.com/img/59a5/3c6f/ae49249b51c7d5d81ab89eeb0bf610f1?Expires=1712534400&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=otR4I8Y0NumUlQW6NsUkXOvTzPqQhjjq1BLNd~EcweuN0Q0vRcVUvMuhlAx2vlsr2lOiqAgoyuXkYlVrK2qknRxQffQDRjGbRsK6CyebC76kXWw5Zu0SxlwtUdiYVV8VI0lWwoRsKqnoI4DXOqChcEMKPQamtpUmTx~NHx8t5cKSdvAMu0tqlPPdF7Sa51Vcuzrryfj~mcZXXEdEltEACAxPsFxhCelyDPB2Se7ZihPK1RGrtvovJZkc-64whNnji8Z0AOm-~irZhl0WQh0jhsaUpp2T5h9drq8-UwVdco3GBNXLSk3ygioYruN0j4U7SkqKVt7~ng1G7IH7395B4A__"
          />
          <styles.percentage>
            <p>50%</p>
          </styles.percentage>
          <p>김마루</p>
        </styles.writer>
      </styles.container>
      <HorizontalDivider />
    </div>
  );
}
