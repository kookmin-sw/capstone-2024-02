'use client';

import styled from 'styled-components';

import { CircularProfileImage } from '@/components';

const styles = {
  container: styled.div`
    width: 16.625rem;
    height: 18.75rem;
    flex-shrink: 0;
    border-radius: 20px;
    background: var(--Gray-1, #f7f6f9);

    padding: 1.88rem 1.25rem;
  `,
  profileInfo: styled.div`
    display: flex;
    align-items: center;
    gap: 1.44rem;

    margin-bottom: 1.25rem;

    div {
      display: flex;
      flex-direction: column;

      h1 {
        color: #000;
        font-family: 'Noto Sans KR';
        font-size: 1.5rem;
        font-style: normal;
        font-weight: 500;
        line-height: normal;
        margin-bottom: 0.5rem;
      }

      p {
        color: #000;
        font-family: 'Noto Sans KR';
        font-size: 1rem;
        font-style: normal;
        font-weight: 500;
        line-height: normal;
        margin-bottom: 0.25rem;
      }
    }
  `,
  data: styled.div`
    display: flex;
    flex-direction: column;
    gap: 0.5rem;

    padding: 0 1.06rem;

    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    .selected {
      color: #e15637;
    }
  `,
};

export function UserCard({
  name,
  percentage,
}: {
  name: string;
  percentage: number;
}) {
  return (
    <styles.container>
      <styles.profileInfo>
        <CircularProfileImage
          diameter={110}
          percentage={percentage}
          url="https://s3-alpha-sig.figma.com/img/59a5/3c6f/ae49249b51c7d5d81ab89eeb0bf610f1?Expires=1712534400&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=otR4I8Y0NumUlQW6NsUkXOvTzPqQhjjq1BLNd~EcweuN0Q0vRcVUvMuhlAx2vlsr2lOiqAgoyuXkYlVrK2qknRxQffQDRjGbRsK6CyebC76kXWw5Zu0SxlwtUdiYVV8VI0lWwoRsKqnoI4DXOqChcEMKPQamtpUmTx~NHx8t5cKSdvAMu0tqlPPdF7Sa51Vcuzrryfj~mcZXXEdEltEACAxPsFxhCelyDPB2Se7ZihPK1RGrtvovJZkc-64whNnji8Z0AOm-~irZhl0WQh0jhsaUpp2T5h9drq8-UwVdco3GBNXLSk3ygioYruN0j4U7SkqKVt7~ng1G7IH7395B4A__"
        />
        <div>
          <h1>{name}</h1>
          <p>24세</p>
          <p>성북 길음동</p>
        </div>
      </styles.profileInfo>
      <styles.data>
        <p className="selected">비흡연</p>
        <p className="selected">아침형</p>
        <p>실내취식 괜찮아요</p>
      </styles.data>
    </styles.container>
  );
}
