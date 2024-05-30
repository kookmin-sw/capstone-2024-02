'use client';

import { useRouter } from 'next/navigation';
import React, { useState, useEffect, useCallback } from 'react';
import styled from 'styled-components';

import { VitalSection, OptionSection } from '@/components';
import { useAuthValue, useUserData } from '@/features/auth';
import { usePutUserCard } from '@/features/profile';
import { useToast } from '@/features/toast';
import Location from '@/public/option-img/location_on.svg';
import Meeting from '@/public/option-img/meeting_room.svg';
import Person from '@/public/option-img/person.svg';
import Visibility from '@/public/option-img/visibility.svg';

const styles = {
  pageContainer: styled.div`
    display: flex;
    width: 100%;
    padding: 0rem 2rem;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #fff;
    gap: 1rem;
    padding: 2rem;
  `,
  contentContainer: styled.div`
    display: flex;
    flex-direction: column;
    gap: 1rem;
  `,
  pageDescription: styled.p`
    align-self: stretch;
    text-align: left;
    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 2rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
  cardContainer: styled.div`
    display: flex;
    align-items: flex-start;
    gap: 1rem;
  `,
  miniCardContainer: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    min-width: 20rem;
    max-width: 20rem;
    gap: 1rem;
    align-self: stretch;
  `,
  cardNameSection: styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    align-self: stretch;
  `,
  miniCard: styled.div<CardActiveProps>`
    display: flex;
    padding: 2rem;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    align-self: stretch;
    border-radius: 1.875rem;
    background: #f7f6f9;

    /* button */
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
    background: ${props =>
      props.$active !== undefined && props.$active !== null && props.$active
        ? 'var(--background, #f7f6f9)'
        : 'var(--White, #fff)'};
    box-shadow: ${props =>
      props.$active !== undefined && props.$active !== null && props.$active
        ? '0px 4px 20px 0px rgba(0, 0, 0, 0.2)'
        : 'none'};
    border: ${props =>
      props.$active !== undefined && props.$active !== null && props.$active
        ? 'none'
        : '2px solid var(--Light-gray, #DCDDEA)'};
  `,
  miniCardName: styled.p<CardActiveProps>`
    color: ${props =>
      props.$active !== undefined && props.$active !== null && props.$active
        ? 'var(--Black, #35373A)'
        : 'var(--Gray-3, #888)'};
    font-family: 'Noto Sans KR';
    font-size: 1.5rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,
  miniCardKeywordsContainer: styled.ul`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1.2rem;
  `,
  miniCardList: styled.li`
    display: flex;
    height: 2rem;
    align-items: center;
    gap: 2rem;
    align-self: stretch;
  `,
  miniCardPerson: styled(Person)<CardActiveProps>`
    width: 1.5rem;
    height: 1.5rem;
    path {
      fill: ${props =>
        props.$active !== undefined && props.$active !== null && props.$active
          ? 'var(--Main-1, #e15637)'
          : 'var(--Main-2, #767D86)'};
    }
  `,
  miniCardLocation: styled(Location)<CardActiveProps>`
    width: 1.5rem;
    height: 1.5rem;
    path {
      fill: ${props =>
        props.$active !== undefined && props.$active !== null && props.$active
          ? 'var(--Main-1, #e15637)'
          : 'var(--Main-2, #767D86)'};
    }
  `,
  miniCardMeeting: styled(Meeting)<CardActiveProps>`
    width: 1.5rem;
    height: 1.5rem;
    path {
      fill: ${props =>
        props.$active !== undefined && props.$active !== null && props.$active
          ? 'var(--Main-1, #e15637)'
          : 'var(--Main-2, #767D86)'};
    }
  `,
  miniCardVisibility: styled(Visibility)<CardActiveProps>`
    width: 1.5rem;
    height: 1.5rem;
    path {
      fill: ${props =>
        props.$active !== undefined && props.$active !== null && props.$active
          ? 'var(--Main-1, #e15637)'
          : 'var(--Main-2, #767D86)'};
    }
  `,
  miniCardText: styled.p<CardActiveProps>`
    flex: 1 0 0;
    display: flex;
    height: 3rem;
    align-items: center;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;

    color: ${props =>
      props.$active !== undefined && props.$active !== null && props.$active
        ? 'var(--Main-1, #e15637)'
        : 'var(--Main-2, #767D86)'};
  `,
  checkSection: styled.div`
    display: flex;
    padding: 2rem;
    flex-direction: column;
    align-items: flex-start;
    gap: 2rem;
    border-radius: 1.875rem;
    background: var(--background, #f7f6f9);
    position: relative;
  `,
  checkContainer: styled.div<CardActiveProps>`
    display: flex;
    padding: 2rem;
    flex-direction: column;
    align-items: flex-start;
    gap: 2rem;
    border-radius: 1.875rem;
    background: var(--background, #f7f6f9);
    position: relative;

    display: ${props =>
      props.$active !== undefined && props.$active !== null && props.$active
        ? ''
        : 'none'};
  `,
  horizontalLine: styled.div`
    height: 0.0625rem;
    background: var(--Gray-9, #d3d0d7);
  `,
  mateButtonContainer: styled.div`
    display: flex;
    padding: 0.75rem 1.5rem;
    justify-content: center;
    align-items: center;
    gap: 0.25rem;
    border-radius: 0.5rem;
    background: var(--Main-1, #e15637);
  `,
  mateButtonDescription: styled.p`
    color: #fff;

    font-family: Pretendard;
    font-size: 1.125rem;
    font-style: normal;
    font-weight: 600;
    line-height: 1.5rem;
  `,
  mateButtonIcon: styled.img`
    width: 1rem;
    height: 1rem;
  `,
};

interface CardActiveProps {
  $active?: boolean;
}

interface UserProps {
  memberId: string | undefined;
  name: string | undefined;
  gender: string | undefined;
  birthYear: string | undefined;
  myCardId: number | undefined;
  mateCardId: number | undefined;
}

const useSelectedState = (): [
  (
    | {
        smoking?: string;
        roomSharingOption?: string;
        mateAge?: number;
        options?: Set<string>;
      }
    | undefined
  ),
  (key: 'smoking' | 'roomSharingOption' | 'mateAge', value: string) => void,
  (option: string) => void,
] => {
  const [features, setFeatures] = useState<{
    smoking?: string;
    roomSharingOption?: string;
    mateAge?: number;
    options?: Set<string>;
  }>({ options: new Set() });

  const handleEssentialFeatureChange = useCallback(
    (
      key: 'smoking' | 'roomSharingOption' | 'mateAge',
      value: string | number,
    ) => {
      setFeatures(prev => {
        if (prev?.[key] === value) {
          return { ...prev, [key]: undefined };
        }
        return { ...prev, [key]: value };
      });
    },
    [],
  );

  const handleOptionalFeatureChange = useCallback((option: string) => {
    setFeatures(prev => {
      const { options } = prev;
      const newOptions = new Set(options);

      if (options != null && options.has(option)) newOptions.delete(option);
      else newOptions.add(option);

      return { ...prev, options: newOptions };
    });
  }, []);

  return [features, handleEssentialFeatureChange, handleOptionalFeatureChange];
};

export function UserInputPage() {
  const auth = useAuthValue();
  const { data } = useUserData(auth?.accessToken !== undefined);
  const [user, setUserData] = useState<UserProps | null>(null);

  const router = useRouter();

  useEffect(() => {
    if (data !== undefined) {
      const { name, gender, birthYear, memberId, myCardId, mateCardId } = data;
      setUserData({ name, gender, birthYear, memberId, myCardId, mateCardId });
    }
  }, [data]);

  const [myFeatures, handleFeatureChange, handleOptionClick] =
    useSelectedState();
  const [mateFeatures, handleMateFeatureChange, handleMateOptionClick] =
    useSelectedState();

  const myCardId = user?.myCardId ?? 0;
  const mateCardId = user?.mateCardId ?? 0;

  const { mutate: mutateMyCard } = usePutUserCard(myCardId);
  const { mutate: mutateMateCard } = usePutUserCard(mateCardId);

  const [locationInput, setLocation] = useState<string | undefined>('');

  const [mbti, setMbti] = useState<string | undefined>('');
  const [major, setMajor] = useState<string | undefined>('');
  const [budget, setBudget] = useState<string | undefined>('');

  const [mateMbti, setMateMbti] = useState<string | undefined>('');
  const [mateMajor, setMateMajor] = useState<string | undefined>('');
  const [mateBudget, setMateBudget] = useState<string | undefined>('');

  const [mateAge, setMateAge] = useState<number | undefined>(0);

  const { createToast } = useToast();

  const handleButtonClick = () => {
    if (locationInput == null || locationInput === '') {
      createToast({
        message: '필수 항목을 입력하셔야 합니다.',
        option: {
          duration: 3000,
        },
      });
      return;
    }
    const location = locationInput ?? '';
    const myOptions: string[] = [mbti ?? '', major ?? '', budget ?? ''];
    myFeatures?.options?.forEach(option => myOptions.push(option));

    const mutateMyfeatures = {
      smoking: myFeatures?.smoking ?? '상관없어요',
      roomSharingOption: myFeatures?.roomSharingOption ?? '상관없어요',
      mateAge: mateAge ?? 0,
      options: JSON.stringify(myOptions.filter(value => value !== '')),
    };

    const mateOptions: string[] = [
      mateMbti ?? '',
      mateMajor ?? '',
      mateBudget ?? '',
    ];
    mateFeatures?.options?.forEach(option => mateOptions.push(option));

    const mutateMateFeatures = {
      smoking: mateFeatures?.smoking ?? '상관없어요',
      roomSharingOption: mateFeatures?.roomSharingOption ?? '상관없어요',
      mateAge: mateAge ?? 0,
      options: JSON.stringify(mateOptions.filter(value => value !== '')),
    };

    try {
      mutateMyCard({
        location,
        features: mutateMyfeatures,
      });
      mutateMateCard({
        location,
        features: mutateMateFeatures,
      });
      router.replace('/');
    } catch (error) {
      console.error(error);
    }
  };

  const [activeContainer, setActiveContainer] = useState<'my' | 'mate'>('my');

  const handleMyCardClick = () => {
    setActiveContainer('my');
  };

  const handleMateCardClick = () => {
    setActiveContainer('mate');
  };

  let ageString;
  switch (mateAge) {
    case 0:
      ageString = '동갑';
      break;
    case undefined:
      ageString = '상관없어요';
      break;
    default:
      ageString = `±${mateAge}년생`;
  }

  return (
    <styles.pageContainer>
      <styles.contentContainer>
        <styles.pageDescription>
          {user?.name} 님과 희망 메이트에 대해서 알려주세요
        </styles.pageDescription>
        <styles.cardContainer>
          <styles.miniCardContainer>
            <styles.cardNameSection>
              <styles.miniCard
                onClick={handleMyCardClick}
                $active={activeContainer === 'my'}
              >
                <styles.miniCardName $active={activeContainer === 'my'}>
                  내 카드
                </styles.miniCardName>
                <styles.miniCardKeywordsContainer>
                  <styles.miniCardList>
                    <styles.miniCardPerson $active={activeContainer === 'my'} />
                    <styles.miniCardText $active={activeContainer === 'my'}>
                      {user?.gender === 'MALE' ? '남성' : '여성'} ·{' '}
                      {user?.birthYear?.slice(2)}년생 · {myFeatures?.smoking}
                    </styles.miniCardText>
                  </styles.miniCardList>
                  <styles.miniCardList>
                    <styles.miniCardLocation
                      $active={activeContainer === 'my'}
                    />
                    <styles.miniCardText $active={activeContainer === 'my'}>
                      {locationInput}
                    </styles.miniCardText>
                  </styles.miniCardList>
                  <styles.miniCardList>
                    <styles.miniCardMeeting
                      $active={activeContainer === 'my'}
                    />
                    <styles.miniCardText $active={activeContainer === 'my'}>
                      메이트와 {myFeatures?.roomSharingOption}
                    </styles.miniCardText>
                  </styles.miniCardList>
                  <styles.miniCardList>
                    <styles.miniCardVisibility
                      $active={activeContainer === 'my'}
                    />
                    <styles.miniCardText $active={activeContainer === 'my'}>
                      {myFeatures?.options != null &&
                      myFeatures.options.has('아침형')
                        ? '아침형'
                        : null}
                      {myFeatures?.options != null &&
                      myFeatures.options.has('올빼미형')
                        ? '올빼미형'
                        : null}
                    </styles.miniCardText>
                  </styles.miniCardList>
                </styles.miniCardKeywordsContainer>
              </styles.miniCard>
              <styles.miniCard
                onClick={handleMateCardClick}
                $active={activeContainer === 'mate'}
              >
                <styles.miniCardName $active={activeContainer === 'mate'}>
                  메이트 카드
                </styles.miniCardName>
                <styles.miniCardKeywordsContainer>
                  <styles.miniCardList>
                    <styles.miniCardPerson
                      $active={activeContainer === 'mate'}
                    />
                    <styles.miniCardText $active={activeContainer === 'mate'}>
                      {user?.gender === 'MALE' ? '남성' : '여성'} · {ageString}{' '}
                      · {mateFeatures?.smoking}
                    </styles.miniCardText>
                  </styles.miniCardList>
                  <styles.miniCardList>
                    <styles.miniCardLocation
                      $active={activeContainer === 'mate'}
                    />
                    <styles.miniCardText $active={activeContainer === 'mate'}>
                      {locationInput}
                    </styles.miniCardText>
                  </styles.miniCardList>
                  <styles.miniCardList>
                    <styles.miniCardMeeting
                      $active={activeContainer === 'mate'}
                    />
                    <styles.miniCardText $active={activeContainer === 'mate'}>
                      메이트와 {mateFeatures?.roomSharingOption}
                    </styles.miniCardText>
                  </styles.miniCardList>
                  <styles.miniCardList>
                    <styles.miniCardVisibility
                      $active={activeContainer === 'mate'}
                    />
                    <styles.miniCardText $active={activeContainer === 'mate'}>
                      {mateFeatures?.options != null &&
                      mateFeatures.options.has('아침형')
                        ? '아침형'
                        : null}
                      {mateFeatures?.options != null &&
                      mateFeatures.options.has('올빼미형')
                        ? '올빼미형'
                        : null}
                    </styles.miniCardText>
                  </styles.miniCardList>
                </styles.miniCardKeywordsContainer>
              </styles.miniCard>
            </styles.cardNameSection>
          </styles.miniCardContainer>
          <styles.checkContainer $active={activeContainer === 'my'}>
            <VitalSection
              gender={user?.gender}
              birthYear={user?.birthYear}
              location={undefined}
              vitalFeatures={undefined}
              onFeatureChange={handleFeatureChange}
              onLocationChange={setLocation}
              onMateAgeChange={() => {}}
              isMySelf
              type="myCard"
            />
            <styles.horizontalLine />
            <OptionSection
              mbti={undefined}
              major={undefined}
              budget={undefined}
              optionFeatures={undefined}
              onFeatureChange={handleOptionClick}
              onMbtiChange={setMbti}
              onMajorChange={setMajor}
              onBudgetChange={setBudget}
              isMySelf
              type="myCard"
            />
          </styles.checkContainer>
          <styles.checkContainer $active={activeContainer === 'mate'}>
            <VitalSection
              gender={user?.gender}
              birthYear={undefined}
              location={locationInput}
              vitalFeatures={undefined}
              onFeatureChange={handleMateFeatureChange}
              onLocationChange={setLocation}
              onMateAgeChange={setMateAge}
              isMySelf
              type="mateCard"
            />
            <styles.horizontalLine />
            <OptionSection
              mbti={undefined}
              major={undefined}
              budget={undefined}
              optionFeatures={undefined}
              onFeatureChange={handleMateOptionClick}
              onMbtiChange={setMateMbti}
              onMajorChange={setMateMajor}
              onBudgetChange={setMateBudget}
              isMySelf
              type="mateCard"
            />
          </styles.checkContainer>
        </styles.cardContainer>
      </styles.contentContainer>
      <styles.mateButtonContainer onClick={handleButtonClick}>
        <styles.mateButtonDescription>
          나의 메이트 확인하기
        </styles.mateButtonDescription>
        <styles.mateButtonIcon src="/chevron-right.svg" />
      </styles.mateButtonContainer>
    </styles.pageContainer>
  );
}
