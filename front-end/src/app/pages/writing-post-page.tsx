'use client';

import { useRouter } from 'next/navigation';
import React, { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';

import {
  LocationSearchBox,
  MateSearchBox,
} from '@/components/writing-post-page';
import { type NaverAddress } from '@/features/geocoding';
import { getImageURL, putImage } from '@/features/image';
import { useCreateSharedPost } from '@/features/shared';

const styles = {
  pageContainer: styled.div`
    display: flex;
    flex-direction: column;
    padding: 2rem 10rem;
    width: 100%;

    max-height: fit-content;

    background: var(--background, #f7f6f9);
  `,
  postContainer: styled.div`
    display: flex;
    padding: 2rem;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    gap: 2rem;
    align-self: stretch;

    border-radius: 16px;
    background: #fff;
  `,
  row: styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    align-self: stretch;
  `,
  option: styled.h2`
    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
    align-self: stretch;
  `,
  captionRow: styled.div`
    display: flex;
    align-items: flex-end;
    gap: 1rem;
    align-self: stretch;
  `,
  caption: styled.span`
    color: rgba(53, 55, 58, 0.5);
    font-family: 'Noto Sans KR';
    font-size: 0.875rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
  `,
  optionRow: styled.div`
    display: flex;
    align-items: flex-start;
    gap: 2rem;
    align-self: stretch;
  `,
  optionCategory: styled.h1`
    align-self: stretch;

    color: #000;
    font-family: 'Noto Sans KR';
    font-size: 1.25rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
  createButton: styled.button`
    all: unset;

    cursor: pointer;

    display: flex;
    width: 7.125rem;
    height: fit-content;
    padding: 0.5rem 1.5rem;
    justify-content: center;
    align-items: center;

    border-radius: 8px;
    background: var(--Black, #35373a);

    color: #fff;
    font-family: Pretendard;
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 600;
    line-height: 1.5rem;
  `,
  titleInput: styled.input`
    all: unset;

    display: flex;
    padding: 1rem;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    align-self: stretch;
    border-radius: 8px;
    background: #ededed;
  `,
  contentInput: styled.input`
    all: unset;

    display: flex;
    padding: 1rem;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    align-self: stretch;
    border-radius: 8px;
    background: #ededed;
  `,
  optionButtonContainer: styled.div`
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;

    span {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 400;
      line-height: normal;
    }
  `,
  customRadioButton: styled.div<ButtonActiveProps>`
    display: inline-block;
    width: 1.5rem;
    height: 1.5rem;
    cursor: pointer;
    background-size: cover;
    ${props =>
      props.$isSelected
        ? {
            backgroundImage: `url('/button-icon/Radio button checked.svg')`,
          }
        : {
            backgroundImage: `url('/button-icon/Radio button unchecked.svg')`,
          }};
  `,
  customCheckBox: styled.div<ButtonActiveProps>`
    display: inline-block;
    width: 1.5rem;
    height: 1.5rem;
    cursor: pointer;
    background-size: cover;
    ${props =>
      props.$isSelected
        ? {
            backgroundImage: `url('/button-icon/Check box.svg')`,
          }
        : {
            backgroundImage: `url('/button-icon/Check box outline blank.svg')`,
          }};
  `,
  images: styled.div`
    display: flex;
    align-items: center;
    align-self: stretch;
    gap: 1rem;

    overflow-x: auto;
  `,
  image: styled.img`
    width: 14.4375rem;
    height: 9.875rem;
    background: #ededed;

    object-fit: cover;
    object-position: center;

    cursor: pointer;
  `,
  imageAddButton: styled.button`
    all: unset;

    border: 0.5px solid #80808080;
    cursor: pointer;

    width: 14.4375rem;
    height: 9.875rem;
    background: #ededed;
    background-image: url(https://s3-alpha-sig.figma.com/img/7307/09fa/b5d93c9ac77c2570ffbee89fe8a76c98?Expires=1714348800&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=Xr~6xrrHnQFb6NwdPlxTuJ2wd7kTnRZ-9kpTlGxYQL-bU7ZVcN8IMQ4k6yEAj~x3y2roX-poLo4XP4x6-adpxlciddzg0ZUuWg0B3VrMgMwbl~sTasgqAe~0SL9E4kkEx7OilanZoC5fJlVBglfb8kE1nZBaG5wEp3FCbLZhzZTnl~29Loisbo1pwteh~2ABpLSVttEztULov1lzws4qcrHY5QpGb8KM4PxBTBTfQDMa8an5QmG~uUlt-bYgVEFMuA2vsKHc-aY8HoiF7v03UDHSGNOVrX1Ajt7ARWqJtOiM~epvCYTkVJPmkNe6WcCgRm37xGKbH2LEzn9aEZJyFA__);
    background-position: center;
    background-repeat: no-repeat;
  `,
  inputContainer: styled.div`
    display: flex;
    width: fit-content;
    padding: 0.5rem;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    border: 1px solid var(--Black, #35373a);
  `,
  input: styled.input<{ $width: number }>`
    all: unset;
    width: ${({ $width }) => `${$width}rem`};
    flex: 1 0 0;
    color: var(--Gray-5, #828282);
    text-align: right;
  `,
  inputPlaceholder: styled.span`
    color: var(--Gray-5, #828282);
    text-align: right;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
  `,
  mates: styled.div`
    display: flex;
    align-items: center;
    align-self: stretch;
    gap: 1rem;

    overflow-x: auto;
  `,
  mate: styled.img`
    width: 5.0625rem;
    height: 5.125rem;
    border-radius: 50%;
    border: 1px solid #dcddea;
    background: #fff;
  `,
  mateAddButton: styled.input`
    all: unset;
    cursor: pointer;

    width: 5.0625rem;
    height: 5.125rem;

    border-radius: 50%;
    border: 1px solid #dcddea;
    background: #fff;
    background-image: url('/icon-plus.png');
    background-position: center;
    background-repeat: no-repeat;
  `,
  address: styled.span`
    color: #e15637;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 400;
    line-height: normal;
  `,
  addressFindButtonContainer: styled.button`
    all: unset;
    cursor: pointer;

    display: flex;
    padding: 0.5rem;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;

    border-radius: 8px;
    background: #ededed;

    span {
      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 400;
      line-height: normal;
    }
  `,
  addressFindButtonIcon: styled.img`
    width: 2rem;
    height: 2rem;
  `,
};

const DealOptions = ['월세', '전세'];
const RoomOptions = ['원룸', '빌라/투룸이상', '아파트', '오피스텔'];
const LivingRoomOptions = ['유', '무'];
const RoomCountOptions = ['1개', '2개', '3개 이상'];
const RestRoomCountOptions = ['1개', '2개', '3개 이상'];
const FloorOptions = ['지상', '반지하', '옥탑'];
const AdditionalOptions = [
  '주차가능',
  '에어컨',
  '냉장고',
  '세탁기',
  '베란다/테라스',
];

interface ButtonActiveProps {
  $isSelected: boolean;
}

interface SelectedOptions {
  budget?: string;
  roomType?: string;
  livingRoom?: string;
  roomCount?: string;
  restRoomCount?: string;
  floorType?: string;
}

type SelectedExtraOptions = Record<string, boolean>;

interface ImageFile {
  url: string;
  file: File;
  extension: string;
}

export function WritingPostPage() {
  const router = useRouter();

  const [images, setImages] = useState<ImageFile[]>([]);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedExtraOptions, setSelectedExtraOptions] =
    useState<SelectedExtraOptions>({});

  const [showMateSearchBox, setShowMateSearchBox] = useState<boolean>(false);
  const [selectedOptions, setSelectedOptions] = useState<SelectedOptions>({});
  const [title, setTitle] = useState<string>('');
  const [content, setContent] = useState<string>('');
  const [mateLimit, setMateLimit] = useState(0);
  const [expectedMonthlyFee, setExpectedMonthlyFee] = useState<number>(0);
  const [houseSize, setHouseSize] = useState<number>(0);

  const [address, setAddress] = useState<NaverAddress | null>(null);
  const [showLocationSearchBox, setShowLocationSearchBox] =
    useState<boolean>(false);

  const { mutate } = useCreateSharedPost();

  const handleExtraOptionClick = (option: string) => {
    setSelectedExtraOptions(prevSelectedOptions => ({
      ...prevSelectedOptions,
      [option]: !prevSelectedOptions[option],
    }));
  };

  const handleOptionClick = (
    optionName: keyof SelectedOptions,
    item: string,
  ) => {
    setSelectedOptions(prevState => ({
      ...prevState,
      [optionName]: prevState[optionName] === item ? null : item,
    }));
  };

  const handleTitleInputChanged = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    setTitle(event.target.value);
  };

  const handleContentInputChanged = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    setContent(event.target.value);
  };

  const handleImageInputClicked = () => {
    imageInputRef.current?.click();
  };

  const handleFileChanged = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { files } = event.target;
    if (files != null) {
      const imagesArray = Array.from(files).map(file => ({
        file,
        url: URL.createObjectURL(file),
        extension: `.${file.type.split('/')[1]}`,
      }));
      setImages(prevImages => [...prevImages, ...imagesArray]);
    }
  };

  const handleRemoveImage = (removeImage: ImageFile) => {
    setImages(prev => prev.filter(image => image.url !== removeImage.url));
  };

  const convertToNumber = (value: string) => {
    const updatedValue = value.replace(/^0+/, '');

    if (updatedValue.length === 0) return 0;

    const numberUpdateValue = Number(updatedValue);
    if (!Number.isNaN(numberUpdateValue)) return numberUpdateValue;
    return null;
  };

  const handleNumberInput = (
    value: string,
    setter: (value: number) => void,
  ) => {
    const converted = convertToNumber(value);
    if (converted == null) return;
    setter(converted);
  };

  const handleCreatePost = (event: React.MouseEvent<HTMLButtonElement>) => {
    (async () => {
      if (images.length > 0) {
        try {
          const getResults = await Promise.allSettled(
            images.map(async ({ extension, file }) => {
              const result = await getImageURL(extension);
              return {
                ...result.data.data,
                file,
              };
            }),
          );

          const urls = getResults.reduce<
            Array<{ file: File; fileName: string; url: string }>
          >((prev, result) => {
            if (result.status === 'rejected') return prev;
            return prev.concat(result.value);
          }, []);

          const putResults = await Promise.allSettled(
            urls.map(async url => {
              await putImage(url.url, url.file);
              return { fileName: url.fileName };
            }),
          );

          const uploadedImages = putResults.reduce<
            Array<{ fileName: string; isThumbNail: boolean; order: number }>
          >((prev, result) => {
            if (result.status === 'rejected') return prev;
            return prev.concat({
              fileName: result.value.fileName,
              isThumbNail: prev.length === 0,
              order: prev.length + 1,
            });
          }, []);

          mutate(
            {
              imageFilesData: uploadedImages,
              postData: { content, title },
              transactionData: {
                rentalType: '0',
                price: 100000,
                monthlyFee: 10000,
                managementFee: 1000,
              },
              roomDetailData: {
                roomType: '0',
                size: 5,
                numberOfRoom: 1,
                recruitmentCapacity: 2,
              },
              locationData: {
                city: 'SEOUL',
                oldAddress: 'test old address',
                roadAddress: 'test road address',
                stationName: 'mokdong',
                stationTime: 10,
                busStopTime: 3,
                schoolName: 'kookmin',
                schoolTime: 20,
                convenienceStoreTime: 2,
              },
            },
            {
              onSuccess: () => {
                router.back();
              },
              onError: () => {},
            },
          );
        } catch (error) {
          console.error(error);
        }
      }
    })().catch((error: Error) => {
      console.error(error);
    });
  };

  useEffect(
    () => () => {
      images.forEach(image => {
        URL.revokeObjectURL(image.url);
      });
    },
    [images],
  );

  return (
    <styles.pageContainer>
      <styles.postContainer>
        <styles.row>
          <styles.optionCategory>기본 정보</styles.optionCategory>
          <styles.createButton onClick={handleCreatePost}>
            작성하기
          </styles.createButton>
        </styles.row>
        <styles.optionCategory>제목</styles.optionCategory>
        <styles.titleInput
          value={title}
          onChange={handleTitleInputChanged}
          placeholder="제목을 입력해주세요"
        />
        <styles.optionCategory>상세 정보</styles.optionCategory>
        <styles.contentInput
          value={content}
          onChange={handleContentInputChanged}
          placeholder="내용을 입력해주세요"
        />
        <styles.captionRow>
          <styles.option>사진</styles.option>
          <styles.caption>최소 2장 이상 업로드</styles.caption>
        </styles.captionRow>
        <styles.images>
          {images.map(image => (
            <styles.image
              key={image.url}
              src={image.url}
              onClick={() => {
                handleRemoveImage(image);
              }}
            />
          ))}
          <styles.imageAddButton onClick={handleImageInputClicked}>
            <input
              ref={imageInputRef}
              type="file"
              multiple
              onChange={handleFileChanged}
              style={{ display: 'none' }}
            />
          </styles.imageAddButton>
        </styles.images>
        <styles.option>인원</styles.option>
        <styles.inputContainer>
          <styles.input
            value={mateLimit}
            onChange={event => {
              handleNumberInput(event.target.value, value => {
                setMateLimit(value);
              });
            }}
            $width={3}
          />
          <styles.inputPlaceholder>명</styles.inputPlaceholder>
        </styles.inputContainer>
        <styles.option>메이트</styles.option>
        <styles.mates>
          <styles.mateAddButton
            onClick={() => {
              setShowMateSearchBox(true);
            }}
          />
          {showMateSearchBox && (
            <MateSearchBox
              setHidden={() => {
                setShowMateSearchBox(false);
              }}
            />
          )}
        </styles.mates>
        <styles.optionCategory>거래 정보</styles.optionCategory>
        <styles.option>거래 방식</styles.option>
        <styles.optionRow>
          {DealOptions.map(option => (
            <styles.optionButtonContainer key={option}>
              <styles.customRadioButton
                $isSelected={selectedOptions.budget === option}
                onClick={() => {
                  handleOptionClick('budget', option);
                }}
              />
              <span>{option}</span>
            </styles.optionButtonContainer>
          ))}
        </styles.optionRow>
        <styles.option>희망 메이트 월 분담금</styles.option>
        <styles.inputContainer>
          <styles.input
            value={expectedMonthlyFee}
            onChange={event => {
              handleNumberInput(event.target.value, value => {
                setExpectedMonthlyFee(value);
              });
            }}
            $width={3}
          />
          <styles.inputPlaceholder>만원</styles.inputPlaceholder>
        </styles.inputContainer>
        <styles.optionCategory>방 정보</styles.optionCategory>
        <styles.option>층</styles.option>
        <styles.optionRow>
          {FloorOptions.map(option => (
            <styles.optionButtonContainer key={option}>
              <styles.customRadioButton
                $isSelected={selectedOptions.floorType === option}
                onClick={() => {
                  handleOptionClick('floorType', option);
                }}
              />
              <span>{option}</span>
            </styles.optionButtonContainer>
          ))}
        </styles.optionRow>
        <styles.option>추가 옵션</styles.option>
        <styles.optionRow>
          {AdditionalOptions.map(option => (
            <styles.optionButtonContainer key={option}>
              <styles.customCheckBox
                $isSelected={selectedExtraOptions[option]}
                onClick={() => {
                  handleExtraOptionClick(option);
                }}
              />
              <span>{option}</span>
            </styles.optionButtonContainer>
          ))}
        </styles.optionRow>
        <styles.option>방 종류</styles.option>
        <styles.optionRow>
          {RoomOptions.map(option => (
            <styles.optionButtonContainer key={option}>
              <styles.customRadioButton
                $isSelected={selectedOptions.roomType === option}
                onClick={() => {
                  handleOptionClick('roomType', option);
                }}
              />
              <span>{option}</span>
            </styles.optionButtonContainer>
          ))}
        </styles.optionRow>
        <styles.option>거실</styles.option>
        <styles.optionRow>
          {LivingRoomOptions.map(option => (
            <styles.optionButtonContainer key={option}>
              <styles.customRadioButton
                $isSelected={selectedOptions.livingRoom === option}
                onClick={() => {
                  handleOptionClick('livingRoom', option);
                }}
              />
              <span>{option}</span>
            </styles.optionButtonContainer>
          ))}
        </styles.optionRow>
        <styles.option>방 개수</styles.option>
        <styles.optionRow>
          {RoomCountOptions.map(option => (
            <styles.optionButtonContainer key={option}>
              <styles.customRadioButton
                $isSelected={selectedOptions.roomCount === option}
                onClick={() => {
                  handleOptionClick('roomCount', option);
                }}
              />
              <span>{option}</span>
            </styles.optionButtonContainer>
          ))}
        </styles.optionRow>
        <styles.option>화장실 개수</styles.option>
        <styles.optionRow>
          {RestRoomCountOptions.map(option => (
            <styles.optionButtonContainer key={option}>
              <styles.customRadioButton
                $isSelected={selectedOptions.restRoomCount === option}
                onClick={() => {
                  handleOptionClick('restRoomCount', option);
                }}
              />
              <span>{option}</span>
            </styles.optionButtonContainer>
          ))}
        </styles.optionRow>
        <styles.option>전체 면적</styles.option>
        <styles.inputContainer>
          <styles.input
            value={houseSize}
            onChange={event => {
              handleNumberInput(event.target.value, value => {
                setHouseSize(value);
              });
            }}
            $width={2}
          />
          <styles.inputPlaceholder>평</styles.inputPlaceholder>
        </styles.inputContainer>
        <styles.optionCategory>위치 정보</styles.optionCategory>
        <styles.captionRow>
          <styles.option>상세 주소</styles.option>
          <styles.address>{address?.roadAddress}</styles.address>
        </styles.captionRow>
        <styles.addressFindButtonContainer
          onClick={() => {
            setShowLocationSearchBox(true);
          }}
        >
          <styles.addressFindButtonIcon src="/icon-search32.svg" />
          <span>위치 찾기</span>
        </styles.addressFindButtonContainer>
        {showLocationSearchBox && (
          <LocationSearchBox
            onSelect={selectedAddress => {
              setAddress(selectedAddress);
            }}
            setHidden={() => {
              setShowLocationSearchBox(false);
            }}
          />
        )}
      </styles.postContainer>
    </styles.pageContainer>
  );
}
