'use client';

import { useRouter } from 'next/navigation';
import styled from 'styled-components';

import { HorizontalDivider } from '@/components';
import {
  type DormitorySharedPostListItem,
  type SharedPostListItem,
} from '@/entities/shared-post';
import { useIsMobile } from '@/shared/mobile';

const styles = {
  container: styled.div`
    width: 100%;
    height: 15rem;

    display: flex;
    gap: 1.56rem;

    @media (max-width: 768px) {
      height: 11rem;
    }
  `,
  thumbnail: styled.img`
    width: 16.125rem;
    height: 12.5rem;

    flex-shrink: 0;
    border-radius: 16px;

    object-fit: cover;

    @media (max-width: 768px) {
      width: 8.5625rem;
      height: 8.625rem;
    }

    cursor: pointer;
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

        cursor: pointer;

        @media (max-width: 768px) {
          font-size: 0.875rem;
        }
      }

      h2 {
        color: var(--Gray-5, #828282);
        font-family: 'Noto Sans KR';
        font-size: 1.125rem;
        font-style: normal;
        font-weight: 400;
        line-height: normal;

        @media (max-width: 768px) {
          font-size: 0.75rem;
        }
      }

      p {
        color: var(--Black, #35373a);
        font-family: 'Noto Sans KR';
        font-size: 1rem;
        font-style: normal;
        font-weight: 500;
        line-height: normal;

        @media (max-width: 768px) {
          font-size: 0.75rem;
        }
      }
    }
  `,
  writer: styled.div`
    position: relative;

    cursor: pointer;

    display: flex;
    flex-shrink: 0;
    flex-direction: column;
    align-items: center;
    gap: 1rem;

    img {
      border-radius: 50%;
      border: 1px solid #dcddea;
      background: #c4c4c4;

      width: 6.25rem;
      height: 6.25rem;
      flex-shrink: 0;
      border-radius: 50%;

      object-fit: cover;

      @media (max-width: 768px) {
        width: 3.375rem;
        height: 3.375rem;
      }
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
    left: 70%;
    top: 25%;

    display: inline-flex;
    padding: 0.75rem 0.375rem;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;

    border-radius: 50%;
    background: #fff;
    box-shadow: 0px 4px 4px 0px rgba(0, 0, 0, 0.04);

    @media (max-width: 768px) {
      padding: 0.5rem 0.2rem;
      top: 40%;
    }

    p {
      color: #e15637;
      font-family: Pretendard;
      font-size: 1rem;
      font-style: normal;
      font-weight: 600;
      line-height: 1.5rem;

      @media (max-width) {
        font-size: 0.625rem;
      }
    }
  `,
};

export function PostCard({
  post,
  onClick,
}: {
  post: SharedPostListItem | DormitorySharedPostListItem;
  onClick: () => void;
}) {
  const router = useRouter();

  const recruitmentCapacity =
    'roomInfo' in post
      ? post.roomInfo.recruitmentCapacity
      : post.recruitmentCapacity;

  const isMobile = useIsMobile();
  return (
    <div>
      <styles.container>
        <styles.thumbnail
          onClick={onClick}
          alt=""
          src={post.thumbnail.fileName}
        />
        <styles.content onClick={onClick}>
          <div>
            <h1>{post.title}</h1>
            <h2>{post.address.roadAddress}</h2>
          </div>
          <div
            style={{
              flexDirection: 'row',
              justifyContent: 'space-between',
              alignItems: 'flex-end',
            }}
          >
            <div>
              <p>모집 {recruitmentCapacity}명</p>
              {'roomInfo' in post && (
                <>
                  <p>
                    {post.roomInfo.roomType} · 방 {post.roomInfo.numberOfRoom} ·
                    화장실 {post.roomInfo.numberOfBathRoom}
                  </p>
                  <p>희망 월 분담금 {post.roomInfo.expectedPayment}만원</p>
                </>
              )}
            </div>
            {isMobile ? (
              <styles.writer
                onClick={() => {
                  router.push(`/profile/${post.publisherAccount.memberId}`);
                }}
              >
                <img alt="" src={post.publisherAccount.profileImageFileName} />
                <styles.percentage>
                  <p>{post.score}</p>
                </styles.percentage>
              </styles.writer>
            ) : null}
          </div>
        </styles.content>
        {!isMobile ? (
          <styles.writer
            onClick={() => {
              router.push(`/profile/${post.publisherAccount.memberId}`);
            }}
          >
            <img alt="" src={post.publisherAccount.profileImageFileName} />
            <styles.percentage>
              <p>{post.score}%</p>
            </styles.percentage>
            <p>{post.publisherAccount.nickname}</p>
          </styles.writer>
        ) : null}
      </styles.container>
      <HorizontalDivider />
    </div>
  );
}
