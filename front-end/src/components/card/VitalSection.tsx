'use client';

import React, { useCallback, useEffect, useState } from 'react';
import styled from 'styled-components';

import { fromAddrToCoord, type NaverAddress } from '@/features/geocoding';
import { useIsMobile } from '@/shared/mobile';

const styles = {
  vitalContainer: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    align-self: stretch;

    @media (max-width: 768px) {
      width: 100%;
    }
  `,
  vitalDescription: styled.p`
    width: 2.6875rem;
    color: var(--Main-1, #e15637);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    @media (max-width: 768px) {
      font-size: 0.75rem;
      width: 1.5rem;
    }
  `,
  vitalListContainer: styled.ul`
    display: flex;
    width: 100%;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    align-self: stretch;
  `,
  vitalList: styled.li`
    display: flex;
    width: 100%;
    align-items: center;
    gap: 2rem;
    align-self: stretch;

    @media (max-width: 768px) {
      gap: 0.1rem;
    }
  `,
  vitalListItemDescription: styled.p`
    width: 5rem;
    color: var(--Main-2, #767d86);
    font-family: 'Noto Sans KR';
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    list-style-type: none;

    @media (max-width: 768px) {
      font-size: 0.625rem;
      width: 3.5rem;
    }
  `,
  vitalCheckListContainer: styled.div`
    display: inline-flex;
    align-items: flex-start;
    gap: 0.5rem;
    flex-wrap: wrap;

    @media (max-width: 768px) {
      gap: 0.2rem;
    }
  `,

  birthYear: styled.select`
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;

    width: 6.7rem;
    height: 3.125rem;
    display: inline-flex;
    padding: 0.75rem 1rem;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    border: 2px solid var(--Gray-4, #dfdfdf);
    background: #fff;

    color: var(--Gray-3, #888);
    font-family: 'Noto Sans KR';
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    background-image: url('/drop-icon.svg');
    background-repeat: no-repeat;
    background-position: calc(100% - 0.6875rem) center;

    padding-right: 2.75rem;

    @media (max-width: 768px) {
      font-size: 0.625rem;
      height: 2.8rem;
      width: 6rem;
    }
  `,

  searchBox: styled.div`
    display: flex;
    padding: 0.875rem 3.0625rem 0.8125rem 1.1875rem;
    align-items: center;
    border: 2px solid var(--Gray-4, #dfdfdf);
    background: var(--White, #fff);
  `,
  mapInput: styled.input`
    color: var(--Gray-3, #888);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
    border: none;
    &:focus {
      outline: none;
    }

    @media (max-width: 768px) {
      font-size: 0.625rem;
    }
  `,
  value: styled.span`
    display: flex;
    padding: 0.5rem 1.5rem;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    border-radius: 10px;
    border: 2px solid #dfdfdf;
    background: #fff;

    color: #888;

    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    @media (max-width: 768px) {
      padding: 0.5rem 1.25rem;
      font-size: 0.625rem;
    }
  `,

  sliderContainer: styled.div`
    width: 50%;
    height: 1.875rem;
    position: relative;
  `,
  sliderTrack: styled.div`
    width: 100%;
    height: 0.3125rem;
    border-radius: 20px;
    background: #d9d9d9;
    position: absolute;
    top: calc(50% - 2px);
  `,
  sliderFillTrack: styled.div<FillProps>`
    width: ${props => props.$fill};
    height: 0.3125rem;
    border-radius: 2px;
    background: var(--Main-1, #e15637);
    position: absolute;
    top: calc(50% - 2px);
  `,
  slider: styled.input`
    position: absolute;
    width: 100%;
    height: 0.3125rem;
    border-radius: 1.25rem;
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    background: transparent;
    top: calc(50% - 2px);

    &:focus {
      outline: none;
    }

    &::-webkit-slider-thumb {
      -webkit-appearance: none;
      pointer-events: all;
      width: 1.875rem;
      height: 1.875rem;
      background-color: #fff;
      border-radius: 50%;
      box-shadow: 0 0 0 1px #c6c6c6;
      cursor: pointer;
      position: relative;
      z-index: 1;
      box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.2);

      @media (max-width: 768px) {
        width: 1.2rem;
        height: 1.2rem;
      }
    }
  `,

  searchAddrContainer: styled.div`
    min-width: 25rem;
    min-height: 10rem;
    display: flex;
    position: absolute;
    left: 9rem;
    top: 12rem;
    flex-direction: column;
    z-index: 200;
    background-color: white;
    padding: 2rem;
    gap: 1rem;
    border-radius: 20px;
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);

    @media (max-width: 768px) {
      top: 10.5rem;
      left: 5.5rem;
      padding: 1rem;
      gap: 0.5rem;
      min-width: 15rem;
    }
  `,
  searchResults: styled.p`
    color: var(--Main-2, #767d86);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    width: 100%;

    @media (max-width: 768px) {
      font-size: 0.625rem;
    }
  `,
};

interface CheckItemProps {
  $isSelected: boolean;
}

interface FillProps {
  $fill: string;
}

const CheckItem = styled.div<CheckItemProps>`
  margin-right: 0.5rem;
  display: flex;
  padding: 0.5rem 1.5rem;
  justify-content: center;
  align-items: center;
  gap: 0.5rem;
  border-radius: 26px;
  border: 2px solid #dfdfdf;
  background: #fff;
  cursor: pointer;

  color: #888;

  font-family: 'Noto Sans KR';
  font-size: 1rem;
  font-style: normal;
  font-weight: 500;
  line-height: normal;

  @media (max-width: 768px) {
    padding: 0.5rem 1rem;
    font-size: 0.625rem;
  }

  ${props =>
    props.$isSelected
      ? {
          color: 'var(--Main-1, #E15637)',
          border: '2px solid var(--Main-1, #E15637)',
        }
      : {
          backgroundColor: '#FFF',
          color: '#888',
        }};
`;

const years = Array.from(
  { length: 100 },
  (_, index) => new Date().getFullYear() - index,
);

export function VitalSection({
  gender,
  birthYear,
  location,
  vitalFeatures,
  onFeatureChange,
  onLocationChange,
  onMateAgeChange,
  isMySelf,
  type,
}: {
  gender?: string;
  birthYear?: string;
  location?: string;
  vitalFeatures?: {
    smoking?: string;
    roomSharingOption?: string;
    mateAge?: number;
    options?: Set<string>;
  };
  onFeatureChange: (
    key: 'smoking' | 'roomSharingOption' | 'mateAge',
    value: string,
  ) => void;
  onLocationChange: React.Dispatch<React.SetStateAction<string | undefined>>;
  onMateAgeChange: React.Dispatch<React.SetStateAction<number | undefined>>;
  isMySelf: boolean;
  type: string;
}) {
  const [features, setFeatures] = useState<{
    smoking?: string;
    roomSharingOption?: string;
    mateAge?: number;
  }>();

  useEffect(() => {
    if (vitalFeatures != null) {
      const data = {
        smoking: vitalFeatures.smoking ?? '상관없어요',
        roomSharingOption: vitalFeatures.roomSharingOption ?? '상관없어요',
        mateAge: vitalFeatures.mateAge ?? 0,
      };

      setFeatures(data);
    }
  }, [vitalFeatures]);

  const handleEssentialFeatureChange = useCallback(
    (key: 'smoking' | 'roomSharingOption' | 'mateAge', value: string) => {
      setFeatures(prev => {
        if (prev?.[key] === value) {
          return { ...prev, [key]: undefined };
        }
        return { ...prev, [key]: value };
      });
      onFeatureChange(key, value);
    },
    [onFeatureChange],
  );

  const [searchText, setSearchText] = useState<string>('');
  const [addresses, setAddresses] = useState<NaverAddress[]>([]);
  const [initialLocation, setInitialLocation] = useState('');
  const [locationBoxClick, setLocationBoxClick] = useState(false);

  useEffect(() => {
    if (location !== undefined && type === 'myCard') {
      setInitialLocation(location);
    }
  }, [location, type]);

  const [locationInput, setLocation] = useState('');
  useEffect(() => {
    setLocation(initialLocation);
  }, [initialLocation]);

  useEffect(() => {
    onLocationChange(locationInput);
  }, [onLocationChange, locationInput]);

  const [initialAge, setInitialAge] = useState<number>(0);
  const [ageValue, setAgeValue] = useState<number>(0);

  useEffect(() => {
    if (vitalFeatures?.mateAge != null) setInitialAge(vitalFeatures.mateAge);
  }, [vitalFeatures?.mateAge]);

  useEffect(() => {
    if (initialAge != null) setAgeValue(initialAge);
  }, [initialAge]);

  const handleAgeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!Number.isNaN(e.target.value)) {
      const n = Number(e.target.value);
      setAgeValue(n);
      onMateAgeChange(n === 11 ? undefined : n);
    }
  };

  let ageValueString;
  switch (ageValue) {
    case 0:
      ageValueString = '동갑';
      break;
    case 11:
      ageValueString = '상관없어요';
      break;
    default:
      ageValueString = `±${ageValue}년생`;
  }

  const isMobile = useIsMobile();

  return (
    <styles.vitalContainer>
      <styles.vitalDescription>필수</styles.vitalDescription>
      <styles.vitalListContainer>
        <styles.vitalList>
          <styles.vitalListItemDescription>
            성별
          </styles.vitalListItemDescription>
          <styles.vitalCheckListContainer>
            <CheckItem
              $isSelected={gender === 'MALE'}
              style={{
                border: gender === 'MALE' ? 'none' : '',
                background: gender === 'MALE' ? 'var(--Gray-5, #828282)' : '',
                color: gender === 'MALE' ? '#fff' : '',
              }}
            >
              남성
            </CheckItem>
            <CheckItem
              $isSelected={gender === 'FEMALE'}
              style={{
                border: gender === 'FEMALE' ? 'none' : '',
                background: gender === 'FEMALE' ? 'var(--Gray-5, #828282)' : '',
                color: gender === 'FEMALE' ? '#fff' : '',
              }}
            >
              여성
            </CheckItem>
          </styles.vitalCheckListContainer>
        </styles.vitalList>
        <styles.vitalList>
          <form
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: isMobile ? '0.1rem' : '2rem',
            }}
            onSubmit={event => {
              event.preventDefault();
              fromAddrToCoord({ query: searchText })
                .then(response => {
                  setAddresses(response.data.addresses);
                })
                .catch((error: Error) => {
                  console.log(error);
                });
            }}
          >
            <styles.vitalListItemDescription>
              희망 지역
            </styles.vitalListItemDescription>
            {type === 'myCard' ? (
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <styles.searchBox>
                  <styles.mapInput
                    readOnly={!isMySelf}
                    placeholder="주소 입력"
                    value={searchText}
                    onChange={event => {
                      setSearchText(event.target.value);
                    }}
                    onClick={() => {
                      setLocationBoxClick(prev => !prev);
                    }}
                  />
                </styles.searchBox>
                {isMobile && locationInput != null ? (
                  <styles.vitalListItemDescription
                    style={{ width: '100%', padding: '0 0.5rem' }}
                  >
                    {locationInput}
                  </styles.vitalListItemDescription>
                ) : null}
              </div>
            ) : (
              <CheckItem
                $isSelected
                style={{
                  border: 'none',
                  background: 'var(--Gray-5, #828282)',
                  color: '#fff',
                }}
              >
                {location}
              </CheckItem>
            )}
          </form>
          {locationBoxClick && (
            <styles.searchAddrContainer>
              {addresses.map(address => (
                <styles.searchResults
                  key={address.roadAddress}
                  onClick={() => {
                    setLocationBoxClick(false);
                    setLocation(address.roadAddress);
                  }}
                >
                  {address.roadAddress}
                </styles.searchResults>
              ))}
            </styles.searchAddrContainer>
          )}
        </styles.vitalList>
        <styles.vitalList>
          <styles.vitalListItemDescription>
            흡연 여부
          </styles.vitalListItemDescription>
          <styles.vitalCheckListContainer>
            <CheckItem
              $isSelected={features?.smoking === '흡연'}
              onClick={() => {
                if (isMySelf) {
                  handleEssentialFeatureChange('smoking', '흡연');
                }
              }}
            >
              흡연
            </CheckItem>
            <CheckItem
              $isSelected={features?.smoking === '비흡연'}
              onClick={() => {
                if (isMySelf) {
                  handleEssentialFeatureChange('smoking', '비흡연');
                }
              }}
            >
              비흡연
            </CheckItem>
            <CheckItem
              $isSelected={features?.smoking === '상관없어요'}
              onClick={() => {
                if (isMySelf) {
                  handleEssentialFeatureChange('smoking', '상관없어요');
                }
              }}
            >
              상관없어요
            </CheckItem>
          </styles.vitalCheckListContainer>
        </styles.vitalList>
        <styles.vitalList>
          <styles.vitalListItemDescription>
            메이트와
          </styles.vitalListItemDescription>
          <styles.vitalCheckListContainer>
            <CheckItem
              $isSelected={features?.roomSharingOption === '같은 방'}
              onClick={() => {
                if (isMySelf) {
                  handleEssentialFeatureChange('roomSharingOption', '같은 방');
                }
              }}
            >
              같은 방
            </CheckItem>
            <CheckItem
              $isSelected={features?.roomSharingOption === '다른 방'}
              onClick={() => {
                if (isMySelf) {
                  handleEssentialFeatureChange('roomSharingOption', '다른 방');
                }
              }}
            >
              다른 방
            </CheckItem>
            <CheckItem
              $isSelected={features?.roomSharingOption === '상관없어요'}
              onClick={() => {
                if (isMySelf) {
                  handleEssentialFeatureChange(
                    'roomSharingOption',
                    '상관없어요',
                  );
                }
              }}
            >
              상관없어요
            </CheckItem>
          </styles.vitalCheckListContainer>
        </styles.vitalList>
        <styles.vitalList>
          <styles.vitalListItemDescription>
            출생 연도
          </styles.vitalListItemDescription>
          {type === 'myCard' ? (
            <styles.birthYear
              value={birthYear}
              disabled={birthYear !== undefined || !isMySelf}
            >
              <option value="">년도</option>
              {years.map(year => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </styles.birthYear>
          ) : (
            <div
              style={{
                display: 'flex',
                gap: '1rem',
                width: '100%',
                alignItems: 'center',
              }}
            >
              <styles.sliderContainer>
                <styles.sliderTrack />
                <styles.sliderFillTrack $fill={`${(ageValue / 11) * 100}%`} />
                <styles.slider
                  type="range"
                  min={0}
                  max={11}
                  step={1}
                  value={ageValue}
                  onChange={handleAgeChange}
                />
              </styles.sliderContainer>
              <styles.value>{ageValueString}</styles.value>
            </div>
          )}
        </styles.vitalList>
      </styles.vitalListContainer>
    </styles.vitalContainer>
  );
}
