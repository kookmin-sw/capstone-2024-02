'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';

import { Bookmark, CircularProfileImage } from '@/components';
import { ImageGrid } from '@/components/shared-post-page';
import { useAuthValue } from '@/features/auth';
import { useCreateChatRoom } from '@/features/chat';
import { fromAddrToCoord } from '@/features/geocoding';
import { useFollowUser, useUnfollowUser } from '@/features/profile';
import {
  useDeleteSharedPost,
  useDormitorySharedPost,
  useScrapSharedPost,
  useSharedPost,
} from '@/features/shared';
import { useToast } from '@/features/toast';
import { getAge } from '@/shared';

const styles = {
  container: styled.div`
    display: flex;
    align-items: center;
    width: 100vw;
    min-width: 390px;
    flex-direction: column;
    gap: 1rem;
    padding: 1rem 0;
  `,
  contentContainer: styled.div`
    display: flex;
    width: 100%;
    justify-content: center;
    align-items: flex-start;
    gap: 1rem;
  `,
  postContainer: styled.div`
    display: flex;
    padding: 0 1rem;
    width: 100%;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;

    border-radius: 16px;
    background: #fff;
  `,
  mateContainer: styled.div`
    display: flex;
    width: 100%;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    gap: 1rem;
  `,
  ImageGrid: styled(ImageGrid)`
    width: 100%;
  `,
  postInfoContainer: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.8rem;

    width: 100%;

    div {
      display: flex;
      justify-content: space-between;
      align-items: center;
      align-self: stretch;

      h1 {
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 1;

        overflow: hidden;
        color: #000;
        text-overflow: ellipsis;
        font-family: 'Noto Sans KR';
        font-size: 1rem;
        font-style: normal;
        font-weight: 700;
        line-height: normal;
      }
    }
  `,
  postInfoContent: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;

    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 0.75rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    & > span {
      width: 100%;
    }

    div {
      display: flex;
      width: 100%;
      justify-content: space-between;
      align-items: center;
    }
  `,
  postContentContainer: styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    gap: 0.8rem;
    align-self: stretch;

    h2 {
      align-self: stretch;

      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
    }

    p {
      align-self: stretch;

      color: var(--Black, #35373a);
      font-family: 'Noto Sans KR';
      font-size: 0.75rem;
      font-style: normal;
      font-weight: 400;
      line-height: normal;
    }
  `,
  divider: styled.div`
    width: 100%;
    height: 0.0625rem;
    background: #d3d0d7;
  `,
  dealInfoContainer: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.8rem;
    align-self: stretch;

    h2 {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
    }

    div {
      display: flex;
      justify-content: space-between;
      align-items: center;
      align-self: stretch;

      color: var(--Black, #35373a);
      font-family: 'Noto Sans KR';
      font-size: 0.75rem;
      font-style: normal;
      font-weight: 400;
      line-height: normal;
    }
  `,
  roomInfoContainer: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.8rem;
    align-self: stretch;

    h2 {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
    }

    div {
      display: flex;
      justify-content: space-between;
      align-items: center;
      align-self: stretch;

      color: var(--Black, #35373a);
      font-family: 'Noto Sans KR';
      font-size: 0.75rem;
      font-style: normal;
      font-weight: 400;
      line-height: normal;
    }
  `,
  locationInfoContainer: styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    gap: 0.8rem;
    align-self: stretch;

    h2 {
      align-self: stretch;

      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
    }

    p {
      align-self: stretch;
      color: var(--Black, #35373a);
      font-family: 'Noto Sans KR';
      font-size: 0.75rem;
      font-style: normal;
      font-weight: 400;
      line-height: normal;
    }

    #map {
      height: 13.375rem;
      align-self: stretch;
    }
  `,
  mates: styled.div`
    display: flex;
    flex-direction: column;
    padding: 1rem;
    align-items: flex-start;

    border-radius: 16px;
    background: #fff;
  `,
  mate: styled.img<{ $selected?: boolean; $zIndex?: number }>`
    cursor: pointer;

    display: flex;
    width: 3.75rem;
    height: 3.75rem;
    justify-content: center;
    align-items: center;
    flex-shrink: 0;

    border-radius: 50%;
    border: 1px solid
      ${({ $selected }) =>
        $selected != null && $selected ? '#e15637' : '#DCDDEA'};
    background: #c4c4c4;

    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);

    z-index: ${({ $zIndex }) => $zIndex};

    &:not(:first-child) {
      margin-top: -1rem;
    }
  `,
  selectedMateContainer: styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    align-self: stretch;

    border-radius: 16px;
    background: #fff;
  `,
  profile: styled.div`
    display: flex;
    gap: 1.25rem;
    align-items: center;
    align-self: stretch;
  `,
  profileInfo: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    gap: 0.5rem;
    align-self: stretch;

    .name {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;
    }

    div {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 0.75rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;
    }
  `,
  buttons: styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    align-self: stretch;

    div {
      display: flex;
      align-items: flex-start;
      gap: 1rem;
      align-self: stretch;
    }
  `,
  chattingButton: styled.button`
    all: unset;
    cursor: pointer;

    display: flex;
    width: 2.125rem;
    padding: 0.25rem 0.75rem;
    justify-content: center;
    align-items: center;
    gap: 0.25rem;

    border-radius: 8px;
    background: var(--Black, #35373a);

    color: #fff;
    font-family: Pretendard;
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 600;
    line-height: 1.5rem;
  `,
  rowForDeleteAndModify: styled.div`
    display: flex;
    flex-direction: row;
    gap: 1rem;

    align-self: end;
  `,
  postModifyButton: styled.button`
    all: unset;
    cursor: pointer;

    display: flex;
    width: fit-content;
    height: fit-content;
    padding: 0.5rem 1.5rem;
    justify-content: center;
    align-items: center;

    border-radius: 8px;
    border: 1px solid var(--Gray-3, #888);
    background: var(--White, #fff);

    color: var(--Gray-3, #888);
    font-family: Pretendard;
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 600;
    line-height: 1.5rem;

    @media (max-width: 768px) {
      font-size: 0.75rem;
      width: 4rem;
      padding: 0.25rem 1rem;
    }
  `,
  postDeleteButton: styled.div`
    all: unset;
    cursor: pointer;

    display: flex;
    width: fit-content;
    height: fit-content;
    padding: 0.5rem 1.5rem;
    justify-content: center;
    align-items: center;

    border-radius: 0.5rem;
    background: #e15637;

    color: #fff;
    text-align: right;
    font-family: Pretendard;
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 600;
    line-height: 1.5rem;

    @media (max-width: 768px) {
      font-size: 0.75rem;
      width: 4rem;
      padding: 0.25rem 1rem;
    }
  `,
};

export function MobileSharedPostPage({
  postId,
  type,
}: {
  postId: number;
  type: 'hasRoom' | 'dormitory';
}) {
  const auth = useAuthValue();

  const [, setMap] = useState<naver.maps.Map | null>(null);
  const mapRef = useRef<HTMLDivElement>(null);

  const router = useRouter();
  const { createToast } = useToast();

  const { mutate: deleteSharedPost } = useDeleteSharedPost();

  const { isLoading: isSharedPostLoading, data: sharedPost } = useSharedPost({
    postId,
    enabled: type === 'hasRoom' && auth?.accessToken != null,
  });

  const { isLoading: isDormitorySharedPostLoading, data: dormitorySharedPost } =
    useDormitorySharedPost({
      postId,
      enabled: type === 'dormitory' && auth?.accessToken != null,
    });

  useEffect(() => {
    if (sharedPost?.data.address.roadAddress != null) {
      fromAddrToCoord({ query: sharedPost?.data.address.roadAddress }).then(
        res => {
          const address = res.data.addresses.shift();
          if (address != null && mapRef.current != null) {
            const center = new naver.maps.LatLng(+address.y, +address.x);
            setMap(
              new naver.maps.Map(mapRef.current, {
                center,
                disableKineticPan: false,
                scrollWheel: false,
              }),
            );
          }
        },
      );
    }
  }, [sharedPost]);

  const { mutate: chattingMutate } = useCreateChatRoom();

  const isLoading = useMemo(
    () =>
      type === 'hasRoom' ? isSharedPostLoading : isDormitorySharedPostLoading,
    [type, isSharedPostLoading, isDormitorySharedPostLoading],
  );

  const post = useMemo(
    () => (type === 'hasRoom' ? sharedPost : dormitorySharedPost),
    [type, sharedPost, dormitorySharedPost],
  );

  const [selected, setSelected] = useState<
    | {
        memberId: string;
        nickname: string;
        profileImageFileName: string;
        birthYear: string;
        isScrapped: boolean;
      }
    | undefined
  >(post != null ? post.data.participants[0] : undefined);

  useEffect(() => {
    if (post == null || selected != null) return;

    setSelected(post.data.participants[0]);
  }, [selected, post]);

  const { mutate: scrapPost } = useScrapSharedPost();

  const { mutate: follow } = useFollowUser(selected?.memberId ?? '');
  const { mutate: unfollow } = useUnfollowUser(selected?.memberId ?? '');

  const [followList, setFollowList] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (post == null) return;

    const { participants } = post.data;
    const newFollowList: Record<string, boolean> = {};
    participants.forEach(({ memberId, isScrapped }) => {
      newFollowList[memberId] = isScrapped;
    });
    setFollowList(newFollowList);
  }, [post]);

  if (isLoading || post == null) return <></>;

  return (
    <styles.container>
      <styles.contentContainer>
        <styles.postContainer>
          <styles.mateContainer>
            <styles.selectedMateContainer>
              <styles.profile>
                <CircularProfileImage
                  diameter={89}
                  percentage={50}
                  url="https://s3-alpha-sig.figma.com/img/59a5/3c6f/ae49249b51c7d5d81ab89eeb0bf610f1?Expires=1714348800&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=Ou47yOoRJ57c0QqtWD~w0S6BP1UYWpmpCOCgsq9YTqfbNq~TmwfAI2T24-fYxpKSiBDv8y1Tkup68OTc5v2ZHIG~~CLwn6NCBF7QqTu7sQB0oPCvdRFdBm~y4wI8VEIErYhPsCuV2k7L0GVlJss4KkeM1tt1RX0kwfINvh03yzFf8wtjd0xsUJjMaKjNxU3muS2Cj8BZymckjgNGrTvafiGbAfHt0Bw2fTkH8tctfNNXpnZgqrEeDldEuENV~g-fSsLSFbMceZGN5ILEd9gd6fnY2YYeB7qtb9xozvczwTbz6kYIzzHJc7veYTsvxjqx~qTiKF2Yrn45cn5pXvOv1w__"
                />
                <styles.profileInfo>
                  <p className="name">{selected?.nickname}</p>
                  <div>
                    <p>
                      {selected?.birthYear != null
                        ? `${getAge(+selected.birthYear)}세`
                        : new Date().getFullYear()}
                    </p>
                  </div>
                </styles.profileInfo>
                <styles.buttons>
                  <div>
                    <Bookmark
                      marked={followList[selected?.memberId ?? ''] ?? false}
                      onToggle={() => {
                        if (
                          selected == null ||
                          followList[selected.memberId] == null
                        )
                          return;
                        if (followList[selected.memberId]) unfollow();
                        else follow();
                      }}
                      hasBorder
                      color="#888"
                    />
                  </div>
                  <styles.chattingButton
                    onClick={() => {
                      if (selected == null) return;

                      chattingMutate({
                        roomName: selected.nickname,
                        members: [selected.memberId],
                      });
                    }}
                  >
                    채팅
                  </styles.chattingButton>
                </styles.buttons>
                <styles.mates>
                  {post.data.participants.map((participant, index) => (
                    <styles.mate
                      key={participant.memberId}
                      $selected={participant.memberId === selected?.memberId}
                      $zIndex={index}
                      src={participant.profileImageFileName}
                      onClick={() => {
                        setSelected(participant);
                      }}
                    />
                  ))}
                </styles.mates>
              </styles.profile>
            </styles.selectedMateContainer>
          </styles.mateContainer>
          <styles.ImageGrid
            images={post.data.roomImages.map(({ fileName }) => fileName)}
          />
          <styles.postInfoContainer>
            <div>
              <h1>{post.data.title}</h1>
              <Bookmark
                hasBorder={false}
                marked={post.data.isScrapped}
                onToggle={() => {
                  scrapPost(postId);
                }}
                color="black"
              />
            </div>
            <styles.postInfoContent>
              {'roomInfo' in post.data && (
                <>
                  <span>모집 {post.data.roomInfo.recruitmentCapacity}명</span>
                  <span>
                    {post.data.roomInfo.roomType} · 방{' '}
                    {post.data.roomInfo.numberOfRoom} · 화장실{' '}
                    {post.data.roomInfo.numberOfBathRoom}
                  </span>
                </>
              )}
              <div>
                {'roomInfo' in post.data && (
                  <span>
                    희망 월 분담금 {post.data.roomInfo.expectedPayment}만원
                  </span>
                )}
                <span>
                  저장 {post.data.scrapCount} · 조회 {post.data.viewCount}
                </span>
              </div>
            </styles.postInfoContent>
          </styles.postInfoContainer>
          <styles.postContentContainer>
            <h2>상세 정보</h2>
            <p>{post.data.content}</p>
          </styles.postContentContainer>
          <styles.divider />
          {'roomInfo' in post.data && (
            <>
              <styles.dealInfoContainer>
                <h2>거래 정보</h2>
                <div>
                  <span>거래 방식</span>
                  <span>{post.data.roomInfo.rentalType}</span>
                </div>
                <div>
                  <span>희망 월 분담금</span>
                  <span>{post.data.roomInfo.expectedPayment}만원</span>
                </div>
              </styles.dealInfoContainer>
              <styles.roomInfoContainer>
                <h2>방 정보</h2>
                <div>
                  <span>방 종류</span>
                  <span>{post.data.roomInfo.roomType}</span>
                </div>
                <div>
                  <span>거실 보유</span>
                  <span>{post.data.roomInfo.hasLivingRoom ? '유' : '무'}</span>
                </div>
                <div>
                  <span>방 개수</span>
                  <span>{post.data.roomInfo.numberOfRoom}개</span>
                </div>
                <div>
                  <span>화장실 개수</span>
                  <span>{post.data.roomInfo.numberOfBathRoom}개</span>
                </div>
                <div>
                  <span>평수</span>
                  <span>{post.data.roomInfo.size}평</span>
                </div>
              </styles.roomInfoContainer>
            </>
          )}
          <styles.locationInfoContainer>
            <h2>위치 정보</h2>
            <p>{post.data.address.roadAddress}</p>
            <div ref={mapRef} id="map" />
          </styles.locationInfoContainer>
          {post.data.publisherAccount.memberId === auth?.user?.memberId && (
            <styles.rowForDeleteAndModify>
              <styles.postModifyButton
                onClick={() => {
                  router.push(
                    `/shared/writing/${type === 'hasRoom' ? 'room' : 'dormitory'}/${post.data.id}`,
                  );
                }}
              >
                수정하기
              </styles.postModifyButton>
              <styles.postDeleteButton
                onClick={() => {
                  deleteSharedPost(postId, {
                    onSuccess: () => {
                      createToast({
                        message: '정상적으로 삭제되었습니다.',
                        option: {
                          duration: 3000,
                        },
                      });
                      router.back();
                    },
                    onError: () => {
                      createToast({
                        message: '삭제하는데 실패하였습니다.',
                        option: {
                          duration: 3000,
                        },
                      });
                    },
                  });
                }}
              >
                삭제하기
              </styles.postDeleteButton>
            </styles.rowForDeleteAndModify>
          )}
        </styles.postContainer>
      </styles.contentContainer>
    </styles.container>
  );
}
