'use client';

import { useCallback } from 'react';
import styled from 'styled-components';

import { RangeSlider } from '@/components/RangeSlider';
import { ToggleSwitch } from '@/components/ToggleSwitch';
import {
  useSharedPostsFilter,
  type FloorType,
  type RoomType,
} from '@/entities/shared-posts-filter';

const styles = {
  container: styled.div`
    display: inline-flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2rem;

    width: 30rem;

    @media (max-width: 768px) {
      gap: 1rem;
      width: 18rem;
    }

    h1 {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1.125rem;
      font-style: normal;
      font-weight: 500;
      line-height: normal;

      @media (max-width: 768px) {
        font-size: 1rem;
      }
    }

    button {
      all: unset;
      cursor: pointer;

      display: flex;
      padding: 0.75rem 1.25rem;
      justify-content: center;
      align-items: center;
      gap: 0.5rem;

      border-radius: 8px;
      border: 1px solid #000;

      @media (max-width: 768px) {
        font-size: 0.75rem;
        padding: 0.5rem 1rem;
      }

      transition:
        150ms border ease-in-out,
        150ms background-color ease-in-out,
        150ms color ease-in-out;
    }

    .selected {
      background-color: #e15637;
      color: white;

      border: 1px solid white;
    }
  `,
  roomType: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    align-self: stretch;

    div {
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
  `,
  livingRoom: styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    align-self: stretch;
  `,
  roomCount: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    align-self: stretch;

    div {
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;
    }
  `,
  restRoomCount: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    align-self: stretch;

    div {
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;
    }
  `,
  size: styled.div`
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
  `,
  floor: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    align-self: stretch;

    div {
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;
    }
  `,
};

export function RoomTypeFilter() {
  const { filter, setFilter } = useSharedPostsFilter();

  const handleRoomTypeClick = useCallback(
    (roomTypeFilterOption: RoomType) => {
      setFilter(prev => {
        const value = prev.roomInfo.roomType?.[roomTypeFilterOption] ?? false;
        return {
          ...prev,
          roomInfo: {
            ...prev.roomInfo,
            roomType: {
              ...prev.roomInfo.roomType,
              [roomTypeFilterOption]: !value,
            },
          },
        };
      });
    },
    [setFilter],
  );
  const isRoomTypeSelected = useCallback(
    (roomTypeFilterOption: RoomType) => {
      if (filter.roomInfo.roomType?.[roomTypeFilterOption] === true)
        return true;
      return false;
    },
    [filter.roomInfo.roomType],
  );

  const handleFloorTypeClick = useCallback(
    (floorTypeFilterOption: FloorType) => {
      setFilter(prev => {
        const value = prev.roomInfo.floor?.[floorTypeFilterOption] ?? false;
        return {
          ...prev,
          roomInfo: {
            ...prev.roomInfo,
            floor: {
              ...prev.roomInfo.floor,
              [floorTypeFilterOption]: !value,
            },
          },
        };
      });
    },
    [setFilter],
  );
  const isFloorTypeSelected = useCallback(
    (floorTypeFilterOption: FloorType) => {
      if (filter.roomInfo.floor?.[floorTypeFilterOption] === true) return true;
      return false;
    },
    [filter.roomInfo.floor],
  );

  return (
    <styles.container>
      <styles.roomType>
        <h1>방 종류</h1>
        <div>
          <button
            type="button"
            className={isRoomTypeSelected('원룸') ? 'selected' : ''}
            onClick={() => {
              handleRoomTypeClick('원룸');
            }}
          >
            원룸
          </button>
          <button
            type="button"
            className={isRoomTypeSelected('빌라/투룸이상') ? 'selected' : ''}
            onClick={() => {
              handleRoomTypeClick('빌라/투룸이상');
            }}
          >
            빌라/투룸이상
          </button>
          <button
            type="button"
            className={isRoomTypeSelected('아파트') ? 'selected' : ''}
            onClick={() => {
              handleRoomTypeClick('아파트');
            }}
          >
            아파트
          </button>
          <button
            type="button"
            className={isRoomTypeSelected('오피스텔') ? 'selected' : ''}
            onClick={() => {
              handleRoomTypeClick('오피스텔');
            }}
          >
            오피스텔
          </button>
        </div>
      </styles.roomType>
      <styles.livingRoom>
        <h1>거실 유무</h1>
        <ToggleSwitch
          isChecked={filter.roomInfo.hasLivingRoom ?? false}
          onToggle={() => {
            setFilter(prev => {
              const value = prev.roomInfo.hasLivingRoom ?? false;
              return {
                ...prev,
                roomInfo: {
                  ...prev.roomInfo,
                  hasLivingRoom: !value,
                },
              };
            });
          }}
        />
      </styles.livingRoom>
      <styles.roomCount>
        <h1>방 개수</h1>
        <div>
          <button
            type="button"
            className={filter.roomInfo?.roomCount === '1개' ? 'selected' : ''}
            onClick={() => {
              setFilter(prev => ({
                ...prev,
                roomInfo: { ...prev.roomInfo, roomCount: '1개' },
              }));
            }}
          >
            1개
          </button>
          <button
            type="button"
            className={filter.roomInfo?.roomCount === '2개' ? 'selected' : ''}
            onClick={() => {
              setFilter(prev => ({
                ...prev,
                roomInfo: { ...prev.roomInfo, roomCount: '2개' },
              }));
            }}
          >
            2개
          </button>
          <button
            type="button"
            className={
              filter.roomInfo?.roomCount === '3개 이상' ? 'selected' : ''
            }
            onClick={() => {
              setFilter(prev => ({
                ...prev,
                roomInfo: { ...prev.roomInfo, roomCount: '3개 이상' },
              }));
            }}
          >
            3개 이상
          </button>
        </div>
      </styles.roomCount>
      <styles.restRoomCount>
        <h1>화장실 개수</h1>
        <div>
          <button
            type="button"
            className={
              filter.roomInfo?.restRoomCount === '1개' ? 'selected' : ''
            }
            onClick={() => {
              setFilter(prev => ({
                ...prev,
                roomInfo: { ...prev.roomInfo, restRoomCount: '1개' },
              }));
            }}
          >
            1개
          </button>
          <button
            type="button"
            className={
              filter.roomInfo?.restRoomCount === '2개' ? 'selected' : ''
            }
            onClick={() => {
              setFilter(prev => ({
                ...prev,
                roomInfo: { ...prev.roomInfo, restRoomCount: '2개' },
              }));
            }}
          >
            2개
          </button>
          <button
            type="button"
            className={
              filter.roomInfo?.restRoomCount === '3개 이상' ? 'selected' : ''
            }
            onClick={() => {
              setFilter(prev => ({
                ...prev,
                roomInfo: { ...prev.roomInfo, restRoomCount: '3개 이상' },
              }));
            }}
          >
            3개 이상
          </button>
        </div>
      </styles.restRoomCount>
      <styles.size>
        <h1>면적</h1>
        <RangeSlider
          min={1}
          max={100}
          step={1}
          onChange={({ low, high }) => {
            setFilter(prev => ({
              ...prev,
              roomInfo: {
                ...prev.roomInfo,
                size: { low, high },
              },
            }));
          }}
        />
      </styles.size>
      <styles.floor>
        <h1>층수</h1>
        <div>
          <button
            type="button"
            className={isFloorTypeSelected('지상') ? 'selected' : ''}
            onClick={() => {
              handleFloorTypeClick('지상');
            }}
          >
            지상
          </button>
          <button
            type="button"
            className={isFloorTypeSelected('반지하') ? 'selected' : ''}
            onClick={() => {
              handleFloorTypeClick('반지하');
            }}
          >
            반지하
          </button>
          <button
            type="button"
            className={isFloorTypeSelected('옥탑') ? 'selected' : ''}
            onClick={() => {
              handleFloorTypeClick('옥탑');
            }}
          >
            옥탑
          </button>
        </div>
      </styles.floor>
    </styles.container>
  );
}
