'use client';

import { useRouter } from 'next/navigation';
import React, { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';

import { UserInputSection } from '@/components';
import {
  LocationSearchBox,
  MateSearchBox,
} from '@/components/writing-post-page';
import {
  AdditionalInfoTypeValue,
  CountTypeValue,
  DealTypeValue,
  FloorTypeValue,
  LivingRoomTypeValue,
  RoomTypeValue,
  type DealType,
  type FloorType,
  type RoomType,
} from '@/entities/shared-posts-filter';
import { useAuthValue } from '@/features/auth';
import { getImageURL, putImage } from '@/features/image';
import {
  useCreateDormitorySharedPost,
  useCreateSharedPost,
  useSharedPostProps,
  useUpdateDormitorySharedPost,
  useUpdateSharedPost,
  type ImageFile,
} from '@/features/shared';
import { useToast } from '@/features/toast';

const styles = {
  pageContainer: styled.div`
    display: flex;
    flex-direction: column;
    padding: 2rem 10rem;
    width: 100%;

    min-height: 100dvh;
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
  essentialInfoContainer: styled.div`
    display: flex;
    flex: 1 0 0;
    width: 100%;
    flex-direction: column;
    gap: 1rem;
  `,
  essentialRow: styled.div`
    display: flex;
    width: 100%;
    flex-direction: row;
    gap: 1rem;

    .column {
      display: flex;
      width: 100%;
      flex-direction: column;
      gap: 1rem;

      overflow-x: auto;
    }
  `,
  mateCardContainer: styled.div`
    display: flex;
    width: 100%;
    flex-direction: column;
    gap: 1rem;

    button {
      all: unset;
      cursor: pointer;

      display: flex;
      width: fit-content;
      padding: 0.5rem 1rem;
      justify-content: center;
      align-items: center;
      gap: 0.5rem;

      border-radius: 0.5rem;
      background: #ededed;

      color: #000;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 400;
      line-height: normal;
    }

    button[class~='edit'] {
      display: flex;
      width: fit-content;
      padding: 0.5rem 1rem;
      justify-content: center;
      align-items: center;
      gap: 0.5rem;

      border-radius: 0.5rem;
      background: #e15637;

      color: #eee;
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 700;
      line-height: normal;
    }
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
    flex-direction: row;
    align-items: flex-end;
    gap: 1rem;
    align-self: stretch;

    .caption {
      color: var(--Black, #35373a);
      font-family: 'Noto Sans KR';
      font-size: 1rem;
      font-style: normal;
      font-weight: 400;
      line-height: normal;
    }
  `,
  dealInfoContainer: styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    gap: 1rem;
    align-self: stretch;
  `,
  roomInfoContainer: styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
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
  contentInput: styled.textarea`
    all: unset;

    display: flex;
    height: 100%;
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
    width: fit-content;
    align-items: center;
    align-self: stretch;
    gap: 1rem;

    overflow-x: auto;
  `,
  image: styled.div<{ $url: string }>`
    width: 14.4375rem;
    height: 9.875rem;
    background: #ededed;

    background-image: ${({ $url }) => `url("${$url}")`};
    background-position: center;
    background-size: cover;

    cursor: pointer;
  `,
  imageAddButton: styled.button`
    all: unset;

    border: 0.5px solid #80808080;
    cursor: pointer;

    width: 14.4375rem;
    height: 9.875rem;
    background: #ededed;
    background-image: url('/icon-plus.png');
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

interface ButtonActiveProps {
  $isSelected: boolean;
}

export function WritingPostPage({
  postId,
  type,
}: {
  postId?: number;
  type: 'hasRoom' | 'dormitory';
}) {
  const router = useRouter();

  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const [showMateSearchBox, setShowMateSearchBox] = useState<boolean>(false);
  const [showMateCardForm, setShowMateCardForm] = useState<boolean>(false);
  const [showLocationSearchBox, setShowLocationSearchBox] =
    useState<boolean>(false);

  const {
    title,
    content,
    images,
    mates,
    mateLimit,
    houseSize,
    address,
    mateCard,
    selectedOptions,
    selectedExtraOptions,
    expectedMonthlyFee,
    derivedMateCardFeatures,
    isCreatable,
    setSharedPostProps,
    handleOptionClick,
    handleExtraOptionClick,
    handleMateCardOptionalFeatureChange,
    handleMateCardEssentialFeatureChange,
    isOptionSelected,
    isExtraOptionSelected,
  } = useSharedPostProps({ postId, type });

  const { mutate: createSharedPost } = useCreateSharedPost();
  const { mutate: updateSharedPost } = useUpdateSharedPost();
  const { mutate: createDormitorySharedPost } = useCreateDormitorySharedPost();
  const { mutate: updateDormitorySharedPost } = useUpdateDormitorySharedPost();

  const { createToast } = useToast();

  const auth = useAuthValue();

  const handleTitleInputChanged = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    setSharedPostProps(prev => ({
      ...prev,
      title: event.target.value,
    }));
  };

  const handleContentInputChanged = (
    event: React.ChangeEvent<HTMLTextAreaElement>,
  ) => {
    setSharedPostProps(prev => ({
      ...prev,
      content: event.target.value,
    }));
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
        uploaded: false,
      }));

      setSharedPostProps(prev => ({
        ...prev,
        images: [...prev.images, ...imagesArray],
      }));
    }
  };

  const handleRemoveImage = (removeImage: ImageFile) => {
    setSharedPostProps(prev => ({
      ...prev,
      images: prev.images.filter(image => image.url !== removeImage.url),
    }));
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
    if (!isCreatable) {
      createToast({
        message: '필수 항목들이 입력되어야 합니다.',
        option: {
          duration: 3000,
        },
      });
      return;
    }

    const extractFileName = (url: string): string => {
      const regex =
        /\/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\.\w+)/;
      const match = url.match(regex);
      return match != null ? match[1] : '';
    };

    const dealType = selectedOptions.budget;
    const { roomType } = selectedOptions;
    const { floorType } = selectedOptions;

    if (
      type === 'hasRoom' &&
      (dealType == null ||
        roomType == null ||
        floorType == null ||
        selectedOptions.roomCount == null ||
        !(selectedOptions.roomCount in CountTypeValue) ||
        selectedOptions.restRoomCount == null ||
        !(selectedOptions.restRoomCount in CountTypeValue))
    )
      return;

    if (address == null || images.length < 2) return;

    const numberOfRoomOption = selectedOptions.roomCount as
      | '1개'
      | '2개'
      | '3개 이상';
    const numberOfRoom = CountTypeValue[numberOfRoomOption];

    const numberOfBathRoomOption = selectedOptions.restRoomCount as
      | '1개'
      | '2개'
      | '3개 이상';
    const numberOfBathRoom = CountTypeValue[numberOfBathRoomOption];

    const dealTypeValue = DealTypeValue[dealType as DealType];
    const roomTypeValue = RoomTypeValue[roomType as RoomType];
    const floorTypeValue = FloorTypeValue[floorType as FloorType];

    (async () => {
      try {
        const getResults = await Promise.allSettled(
          images.map(async ({ url, extension, file, uploaded }) => {
            if (uploaded || extension == null) return { url, uploaded };
            const result = await getImageURL(extension);
            return {
              ...result.data.data,
              uploaded,
              file,
            };
          }),
        );

        const urls = getResults.reduce<
          Array<{
            file?: File;
            fileName?: string;
            url: string;
            uploaded: boolean;
          }>
        >((prev, result) => {
          if (result.status === 'rejected') return prev;
          return prev.concat(result.value);
        }, []);

        const putResults = await Promise.allSettled(
          urls.map(async ({ url, fileName, file, uploaded }) => {
            if (uploaded) return { uploaded, fileName: url };

            if (file != null) await putImage(url, file);
            return { uploaded, fileName };
          }),
        );

        const uploadedImages = putResults.reduce<
          Array<{ fileName: string; isThumbNail: boolean; order: number }>
        >((prev, result) => {
          if (result.status === 'rejected' || result.value.fileName == null)
            return prev;
          const { uploaded, fileName } = result.value;
          return prev.concat({
            fileName: uploaded
              ? `images/${extractFileName(fileName)}`
              : fileName,
            isThumbNail: prev.length === 0,
            order: prev.length + 1,
          });
        }, []);

        if (type === 'hasRoom') {
          if (postId == null) {
            createSharedPost(
              {
                imageFilesData: uploadedImages,
                postData: { title, content },
                transactionData: {
                  rentalType: dealTypeValue,
                  expectedPayment: expectedMonthlyFee,
                },
                roomDetailData: {
                  roomType: roomTypeValue,
                  floorType: floorTypeValue,
                  size: houseSize,
                  numberOfRoom,
                  numberOfBathRoom,
                  hasLivingRoom: selectedOptions.livingRoom === '유',
                  recruitmentCapacity: mateLimit,
                  extraOption: {
                    canPark: selectedExtraOptions.canPark ?? false,
                    hasAirConditioner:
                      selectedExtraOptions.hasAirConditioner ?? false,
                    hasRefrigerator:
                      selectedExtraOptions.hasRefrigerator ?? false,
                    hasWasher: selectedExtraOptions.hasWasher ?? false,
                    hasTerrace: selectedExtraOptions.hasTerrace ?? false,
                  },
                },
                locationData: {
                  oldAddress: address.jibunAddress,
                  roadAddress: address.roadAddress,
                },
                roomMateCardData: {
                  location: address.roadAddress,
                  features: derivedMateCardFeatures,
                },
                participationData: {
                  recruitmentCapacity: mateLimit,
                  participationMemberIds:
                    auth?.user != null
                      ? [
                          auth.user.memberId,
                          ...Object.values(mates).map(mate => mate.memberId),
                        ]
                      : [],
                },
              },
              {
                onSuccess: () => {
                  createToast({
                    message: '게시글이 정상적으로 업로드되었습니다.',
                    option: {
                      duration: 3000,
                    },
                  });
                  router.back();
                },
                onError: () => {
                  createToast({
                    message: '게시글 업로드에 실패했습니다.',
                    option: {
                      duration: 3000,
                    },
                  });
                },
              },
            );
          } else if (postId != null) {
            updateSharedPost(
              {
                postId,
                postData: {
                  imageFilesData: uploadedImages,
                  postData: { title, content },
                  transactionData: {
                    rentalType: dealTypeValue,
                    expectedPayment: expectedMonthlyFee,
                  },
                  roomDetailData: {
                    roomType: roomTypeValue,
                    floorType: floorTypeValue,
                    size: houseSize,
                    numberOfRoom,
                    numberOfBathRoom,
                    hasLivingRoom: selectedOptions.livingRoom === '유',
                    recruitmentCapacity: mateLimit,
                    extraOption: {
                      canPark: selectedExtraOptions.canPark ?? false,
                      hasAirConditioner:
                        selectedExtraOptions.hasAirConditioner ?? false,
                      hasRefrigerator:
                        selectedExtraOptions.hasRefrigerator ?? false,
                      hasWasher: selectedExtraOptions.hasWasher ?? false,
                      hasTerrace: selectedExtraOptions.hasTerrace ?? false,
                    },
                  },
                  locationData: {
                    oldAddress: address.jibunAddress,
                    roadAddress: address.roadAddress,
                  },
                  roomMateCardData: {
                    location: address.roadAddress,
                    features: derivedMateCardFeatures,
                  },
                  participationData: {
                    recruitmentCapacity: mateLimit,
                    participationMemberIds:
                      auth?.user != null
                        ? [
                            auth.user.memberId,
                            ...Object.values(mates).map(mate => mate.memberId),
                          ]
                        : [],
                  },
                },
              },
              {
                onSuccess: () => {
                  createToast({
                    message: '게시글이 정상적으로 수정되었습니다.',
                    option: {
                      duration: 3000,
                    },
                  });
                  router.back();
                },
                onError: () => {
                  createToast({
                    message: '게시글 수정에 실패했습니다.',
                    option: {
                      duration: 3000,
                    },
                  });
                },
              },
            );
          }
        } else if (type === 'dormitory') {
          if (postId == null) {
            createDormitorySharedPost(
              {
                imageFilesData: uploadedImages,
                postData: { title, content },
                locationData: {
                  oldAddress: address.jibunAddress,
                  roadAddress: address.roadAddress,
                },
                roomMateCardData: {
                  location: address.roadAddress,
                  features: derivedMateCardFeatures,
                },
                participationData: {
                  recruitmentCapacity: mateLimit,
                  participationMemberIds:
                    auth?.user != null
                      ? [
                          auth.user.memberId,
                          ...Object.values(mates).map(mate => mate.memberId),
                        ]
                      : [],
                },
              },
              {
                onSuccess: () => {
                  createToast({
                    message: '게시글이 정상적으로 업로드되었습니다.',
                    option: {
                      duration: 3000,
                    },
                  });
                  router.back();
                },
                onError: () => {
                  createToast({
                    message: '게시글 업로드에 실패했습니다.',
                    option: {
                      duration: 3000,
                    },
                  });
                },
              },
            );
          } else if (postId != null) {
            updateDormitorySharedPost(
              {
                postId,
                postData: {
                  imageFilesData: uploadedImages,
                  postData: { title, content },
                  locationData: {
                    oldAddress: address.jibunAddress,
                    roadAddress: address.roadAddress,
                  },
                  roomMateCardData: {
                    location: address.roadAddress,
                    features: derivedMateCardFeatures,
                  },
                  participationData: {
                    recruitmentCapacity: mateLimit,
                    participationMemberIds:
                      auth?.user != null
                        ? [
                            auth.user.memberId,
                            ...Object.values(mates).map(mate => mate.memberId),
                          ]
                        : [],
                  },
                },
              },
              {
                onSuccess: () => {
                  createToast({
                    message: '게시글이 정상적으로 수정되었습니다.',
                    option: {
                      duration: 3000,
                    },
                  });
                  router.back();
                },
                onError: () => {
                  createToast({
                    message: '게시글 수정에 실패했습니다.',
                    option: {
                      duration: 3000,
                    },
                  });
                },
              },
            );
          }
        }
      } catch (error) {
        createToast({
          message: '게시글 업로드에 실패했습니다.',
          option: {
            duration: 3000,
          },
        });
      }
    })();
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
        <styles.essentialInfoContainer>
          <styles.row>
            <styles.optionCategory>기본 정보</styles.optionCategory>
            <styles.createButton onClick={handleCreatePost}>
              {postId == null ? '작성하기' : '수정하기'}
            </styles.createButton>
          </styles.row>
          <styles.optionCategory>제목</styles.optionCategory>
          <styles.titleInput
            value={title}
            onChange={handleTitleInputChanged}
            placeholder="제목을 입력해주세요"
          />
          <styles.optionCategory>위치 정보</styles.optionCategory>
          <styles.captionRow>
            <styles.addressFindButtonContainer
              onClick={() => {
                setShowLocationSearchBox(true);
              }}
            >
              <styles.addressFindButtonIcon src="/icon-search32.svg" />
              <span>위치 찾기</span>
            </styles.addressFindButtonContainer>
            <span className="caption" style={{ alignSelf: 'center' }}>
              상세 주소:
            </span>
            <styles.address style={{ alignSelf: 'center' }}>
              {address?.roadAddress ?? '주소를 입력해주세요.'}
            </styles.address>
          </styles.captionRow>
          {showLocationSearchBox && (
            <LocationSearchBox
              onSelect={selectedAddress => {
                setSharedPostProps(prev => ({
                  ...prev,
                  address: selectedAddress,
                }));
                setShowLocationSearchBox(false);
              }}
              setHidden={() => {
                setShowLocationSearchBox(false);
              }}
            />
          )}
          <styles.essentialRow>
            <div className="column">
              <styles.option>상세 정보</styles.option>
              <styles.contentInput
                value={content}
                onChange={handleContentInputChanged}
                placeholder="내용을 입력해주세요"
              />
            </div>
            <div className="column">
              <styles.captionRow>
                <styles.option>사진</styles.option>
                <styles.caption>최소 2장 이상 업로드</styles.caption>
              </styles.captionRow>
              <styles.images>
                {images.map(image => (
                  <styles.image
                    key={image.url}
                    $url={image.url}
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
            </div>
          </styles.essentialRow>
          <styles.essentialRow>
            <div className="column">
              <styles.option>모집 할 인원</styles.option>
              <styles.inputContainer>
                <styles.input
                  value={mateLimit}
                  onChange={event => {
                    handleNumberInput(event.target.value, value => {
                      setSharedPostProps(prev => ({
                        ...prev,
                        mateLimit: value,
                      }));
                    });
                  }}
                  $width={3}
                />
                <styles.inputPlaceholder>명</styles.inputPlaceholder>
              </styles.inputContainer>
            </div>
            <div className="column">
              <styles.option>메이트</styles.option>
              <styles.mates>
                {Object.values(mates).map(user => (
                  <styles.mate
                    key={user.memberId}
                    alt="Profile"
                    src={user.profileImage}
                  />
                ))}
                <styles.mateAddButton
                  onClick={() => {
                    setShowMateSearchBox(true);
                  }}
                />
                {showMateSearchBox && (
                  <MateSearchBox
                    selectedMates={
                      new Set(Object.values(mates).map(mate => mate.memberId))
                    }
                    onMateSelected={user => {
                      setSharedPostProps(prev => {
                        const next = { ...prev.mates };

                        if (user.memberId in next) {
                          const { [user.memberId]: _, ...rest } = next;
                          return { ...prev, mates: rest };
                        }

                        return {
                          ...prev,
                          mates: { ...next, [user.memberId]: user },
                        };
                      });
                    }}
                    setHidden={() => {
                      setShowMateSearchBox(false);
                    }}
                  />
                )}
              </styles.mates>
            </div>
          </styles.essentialRow>
          <styles.mateCardContainer>
            <styles.option>메이트 카드</styles.option>
            <button
              type="button"
              onClick={() => {
                if (address != null) {
                  setShowMateCardForm(prev => !prev);
                } else {
                  createToast({
                    message: '주소를 먼저 입력해주세요',
                    option: { duration: 3000 },
                  });
                }
              }}
            >
              메이트 카드 작성하기
            </button>
            {showMateCardForm && (
              <UserInputSection
                gender={mateCard.gender}
                birthYear={mateCard.birthYear?.toString()}
                location={address?.roadAddress ?? '주소를 입력해주세요.'}
                mbti={mateCard.mbti}
                major={mateCard.major}
                budget={mateCard.budget}
                features={mateCard.features}
                isMySelf
                type="mateCard"
                onVitalChange={handleMateCardEssentialFeatureChange}
                onOptionChange={handleMateCardOptionalFeatureChange}
                onLocationChange={() => {}}
                onMateAgeChange={mateAge => {
                  setSharedPostProps(prev => ({
                    ...prev,
                    mateCard: {
                      ...prev.mateCard,
                      features: { ...prev.mateCard.features, mateAge },
                    },
                  }));
                }}
                onMbtiChange={() => {}}
                onMajorChange={() => {}}
                onBudgetChange={() => {}}
              />
            )}
          </styles.mateCardContainer>
        </styles.essentialInfoContainer>
        {type === 'hasRoom' && (
          <>
            <styles.dealInfoContainer>
              <styles.optionCategory>거래 정보</styles.optionCategory>
              <styles.option>거래 방식</styles.option>
              <styles.optionRow>
                {Object.keys(DealTypeValue).map(option => (
                  <styles.optionButtonContainer key={option}>
                    <styles.customRadioButton
                      $isSelected={isOptionSelected('budget', option)}
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
                      setSharedPostProps(prev => ({
                        ...prev,
                        expectedMonthlyFee: value,
                      }));
                    });
                  }}
                  $width={3}
                />
                <styles.inputPlaceholder>만원</styles.inputPlaceholder>
              </styles.inputContainer>
            </styles.dealInfoContainer>
            <styles.roomInfoContainer>
              <styles.optionCategory>방 정보</styles.optionCategory>
              <styles.option>층</styles.option>
              <styles.optionRow>
                {Object.keys(FloorTypeValue).map(option => (
                  <styles.optionButtonContainer key={option}>
                    <styles.customRadioButton
                      $isSelected={isOptionSelected('floorType', option)}
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
                {Object.entries(AdditionalInfoTypeValue).map(
                  ([option, value]) => (
                    <styles.optionButtonContainer key={option}>
                      <styles.customCheckBox
                        $isSelected={isExtraOptionSelected(value)}
                        onClick={() => {
                          handleExtraOptionClick(value);
                        }}
                      />
                      <span>{option}</span>
                    </styles.optionButtonContainer>
                  ),
                )}
              </styles.optionRow>
              <styles.option>방 종류</styles.option>
              <styles.optionRow>
                {Object.keys(RoomTypeValue).map(option => (
                  <styles.optionButtonContainer key={option}>
                    <styles.customRadioButton
                      $isSelected={isOptionSelected('roomType', option)}
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
                {Object.keys(LivingRoomTypeValue).map(option => (
                  <styles.optionButtonContainer key={option}>
                    <styles.customRadioButton
                      $isSelected={isOptionSelected('livingRoom', option)}
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
                {Object.keys(CountTypeValue).map(option => (
                  <styles.optionButtonContainer key={option}>
                    <styles.customRadioButton
                      $isSelected={isOptionSelected('roomCount', option)}
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
                {Object.keys(CountTypeValue).map(option => (
                  <styles.optionButtonContainer key={option}>
                    <styles.customRadioButton
                      $isSelected={isOptionSelected('restRoomCount', option)}
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
                      setSharedPostProps(prev => ({
                        ...prev,
                        houseSize: value,
                      }));
                    });
                  }}
                  $width={2}
                />
                <styles.inputPlaceholder>평</styles.inputPlaceholder>
              </styles.inputContainer>
            </styles.roomInfoContainer>
          </>
        )}
      </styles.postContainer>
    </styles.pageContainer>
  );
}
