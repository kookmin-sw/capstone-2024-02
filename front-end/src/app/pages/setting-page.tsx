'use client';

import { type NavigateOptions } from 'next/dist/shared/lib/app-router-context.shared-runtime';
import { useSearchParams, useRouter } from 'next/navigation';
import { useEffect, useState, useRef, useCallback } from 'react';
import styled from 'styled-components';

import { UserInputSection } from '@/components';
import { useAuthValue } from '@/features/auth';
import {
  usePutUserCard,
  useUserCard,
  useUserProfile,
} from '@/features/profile';
import Location from '@/public/option-img/location_on.svg';
import Meeting from '@/public/option-img/meeting_room.svg';
import Person from '@/public/option-img/person.svg';
import Visibility from '@/public/option-img/visibility.svg';

const styles = {
  pageContainer: styled.div`
    display: flex;
    width: 90rem;
    height: 90rem;
    padding: 2rem 10rem;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 2rem;
  `,

  cardName: styled.p`
    align-self: stretch;
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
  miniCard: styled.div`
    display: flex;
    width: 22.5rem;
    padding: 2rem;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    border-radius: 1.875rem;
    background: #f7f6f9;

    /* button */
    box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
  `,
  miniCardName: styled.p`
    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 1.5rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  `,
  miniCardKeywordsContainer: styled.ul`
    display: flex;
    width: 18.375rem;
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
  miniCardPerson: styled(Person)`
    width: 1.5rem;
    height: 1.5rem;
    path {
      fill: var(--Main-1, #e15637);
    }
  `,
  miniCardLocation: styled(Location)`
    width: 1.5rem;
    height: 1.5rem;
    path {
      fill: var(--Main-1, #e15637);
    }
  `,
  miniCardMeeting: styled(Meeting)`
    width: 1.5rem;
    height: 1.5rem;
    path {
      fill: var(--Main-1, #e15637);
    }
  `,
  miniCardVisibility: styled(Visibility)`
    width: 1.5rem;
    height: 1.5rem;
    path {
      fill: var(--Main-1, #e15637);
    }
  `,
  miniCardText: styled.p`
    flex: 1 0 0;
    display: flex;
    height: 3rem;
    align-items: center;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    color: var(--Main-1, #e15637);
  `,
  checkContainer: styled.div`
    display: flex;
    width: 50rem;
    padding: 2rem;
    flex-direction: column;
    align-items: flex-start;
    gap: 2rem;
    border-radius: 1.875rem;
    background: var(--background, #f7f6f9);
  `,
  horizontalLine: styled.div`
    width: 43.75rem;
    height: 0.0625rem;
    background: var(--Gray-9, #d3d0d7);
  `,
};

interface UserProps {
  name: string;
  gender: string;
  birthYear: string;
}

export function SettingPage({ cardId }: { cardId: number }) {
  const params = useSearchParams();
  const memberIdParams = params.get('memberId');
  const memberId = memberIdParams ?? '';
  const isMySelfStr = params.get('isMySelf');
  const isMySelf = isMySelfStr === 'true';
  const type = params.get('type') ?? '';

  const auth = useAuthValue();
  const { mutate: mutateProfile, data: profileData } = useUserProfile(memberId);
  const [userData, setUserData] = useState<UserProps | null>(null);

  useEffect(() => {
    mutateProfile();
  }, [auth]);

  useEffect(() => {
    if (profileData?.data !== undefined) {
      const userProfileData = profileData.data.authResponse;
      if (userProfileData !== undefined) {
        const { name, birthYear, gender } = userProfileData;
        setUserData({ name, gender, birthYear });
      }
    }
  }, [profileData]);

  const card = useUserCard(cardId);

  const [locationInput, setLocation] = useState<string | undefined>(
    card.data?.data.location,
  );

  useEffect(() => {
    if (card.data?.data.location !== undefined) {
      setLocation(card.data.data.location);
    }
  }, [card.data?.data.location]);

  const [features, setFeatures] = useState<{
    smoking?: string;
    roomSharingOption?: string;
    mateAge?: number;
    options?: Set<string>;
  }>({ options: new Set() });

  const [initialMbti, setInitialMbti] = useState('');
  const [initialMajor, setInitialMajor] = useState('');
  const [initialBudget, setInitialBudget] = useState('');

  const majorArray = ['공학', '교육', '인문', '사회', '자연', '예체능', '의약'];

  const [mbti, setMbti] = useState<string | undefined>('');
  const [major, setMajor] = useState<string | undefined>('');
  const [budget, setBudget] = useState<string | undefined>('');
  const [mateAge, setMateAge] = useState<number | undefined>(0);

  useEffect(() => {
    if (isMySelf) {
      if (card != null) {
        const featuresData = card.data?.data.myFeatures;
        if (featuresData != null) {
          const options = JSON.parse(featuresData.options);
          const optionsSet = new Set<string>();
          options.forEach((option: string) => {
            if (
              !option.includes('E') &&
              !option.includes('I') &&
              !option.includes(',') &&
              !majorArray.includes(option)
            ) {
              optionsSet.add(option);
            }
            if (option.includes('E') || option.includes('I'))
              setInitialMbti(option);
            if (option.includes(',')) {
              setInitialBudget(option);
            }
            if (majorArray.includes(option)) setInitialMajor(option);
          });
          const data = {
            smoking: featuresData.smoking,
            roomSharingOption: featuresData.roomSharingOption,
            mateAge: featuresData.mateAge,
            options: optionsSet,
          };
          setMateAge(featuresData.mateAge);
          setFeatures(data);
        }
      }
    }
  }, [isMySelf, card.data?.data.myFeatures]);

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

      if (options != null && options.has(option)) {
        newOptions.delete(option);
        console.log(newOptions);
      } else newOptions.add(option);

      return { ...prev, options: newOptions };
    });
  }, []);

  const { mutate } = usePutUserCard(cardId);
  const router = useRouter();

  const saveData = () => {
    const location = locationInput ?? '';
    const options: string[] = [mbti ?? '', major ?? '', budget ?? ''];
    features?.options?.forEach(option => options.push(option));

    const myFeatures = {
      smoking: features?.smoking ?? '상관없어요',
      roomSharingOption: features?.roomSharingOption ?? '상관없어요',
      mateAge: mateAge ?? 0,
      options: JSON.stringify(options.filter(value => value !== '')),
    };

    mutate({ location, features: myFeatures });
  };

  const handleBeforeUnload = () => {
    saveData();
  };

  const isClickedFirst = useRef(false);

  const handlePopState = () => {
    saveData();
    history.back();
  };

  useEffect(() => {
    const originalPush = router.push.bind(router);
    const newPush = (
      href: string,
      options?: NavigateOptions | undefined,
    ): void => {
      saveData();
      originalPush(href, options);
    };
    router.push = newPush;
    window.onbeforeunload = handleBeforeUnload;
    return () => {
      router.push = originalPush;
      window.onbeforeunload = null;
    };
  }, [router, handleBeforeUnload]);

  useEffect(() => {
    if (!isClickedFirst.current) {
      history.pushState(null, '', '');
      isClickedFirst.current = true;
    }
  }, []);

  useEffect(() => {
    window.addEventListener('popstate', handlePopState);
    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, [handlePopState]);

  let ageString: string;

  if (type === 'myCard') {
    ageString = `${userData?.birthYear.slice(2)}년생`;
  } else {
    switch (mateAge) {
      case 0:
        ageString = '동갑';
        break;
      case 11:
        ageString = '상관없어요';
        break;
      default:
        ageString = `±${mateAge}년생`;
    }
  }

  return (
    <styles.pageContainer>
      <styles.cardName>
        {type === 'myCard' ? `마이 카드` : '메이트 카드'}
      </styles.cardName>
      <styles.cardContainer>
        <styles.miniCard>
          <styles.miniCardName>
            {type === 'myCard' ? '내카드' : '메이트카드'}
          </styles.miniCardName>
          <styles.miniCardKeywordsContainer>
            <styles.miniCardList>
              <styles.miniCardPerson />
              <styles.miniCardText>
                {userData?.gender === 'MALE' ? '남성' : '여성'} · {ageString} ·{' '}
                {features?.smoking}
              </styles.miniCardText>
            </styles.miniCardList>
            <styles.miniCardList>
              <styles.miniCardLocation />
              <styles.miniCardText>{locationInput}</styles.miniCardText>
            </styles.miniCardList>
            <styles.miniCardList>
              <styles.miniCardMeeting />
              <styles.miniCardText>
                메이트와 {features?.roomSharingOption}
              </styles.miniCardText>
            </styles.miniCardList>
            <styles.miniCardList>
              <styles.miniCardVisibility />
              <styles.miniCardText>
                {features?.options != null && features.options.has('아침형')
                  ? '아침형'
                  : null}
                {features?.options != null && features.options.has('올빼미형')
                  ? '올빼미형'
                  : null}
              </styles.miniCardText>
            </styles.miniCardList>
          </styles.miniCardKeywordsContainer>
        </styles.miniCard>
        {type === 'myCard' ? (
          <UserInputSection
            gender={userData?.gender}
            birthYear={userData?.birthYear}
            location={card.data?.data.location}
            features={features}
            isMySelf={isMySelf}
            type="myCard"
            mbti={initialMbti}
            major={initialMajor}
            budget={initialBudget}
            onVitalChange={handleEssentialFeatureChange}
            onOptionChange={handleOptionalFeatureChange}
            onLocationChange={setLocation}
            onMateAgeChange={setMateAge}
            onMbtiChange={setMbti}
            onMajorChange={setMajor}
            onBudgetChange={setBudget}
          />
        ) : (
          <UserInputSection
            gender={userData?.gender}
            birthYear={userData?.birthYear}
            location={card.data?.data.location}
            features={features}
            isMySelf={isMySelf}
            type="mateCard"
            mbti={initialMbti}
            major={initialMajor}
            budget={initialBudget}
            onVitalChange={handleEssentialFeatureChange}
            onOptionChange={handleOptionalFeatureChange}
            onLocationChange={setLocation}
            onMateAgeChange={setMateAge}
            onMbtiChange={setMbti}
            onMajorChange={setMajor}
            onBudgetChange={setBudget}
          />
        )}
      </styles.cardContainer>
    </styles.pageContainer>
  );
}
