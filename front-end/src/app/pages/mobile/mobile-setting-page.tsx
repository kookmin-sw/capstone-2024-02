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

const styles = {
  pageContainer: styled.div`
    display: flex;
    align-items: center;
    width: 100vw;
    min-width: 390px;
    flex-direction: column;
    gap: 1rem;
    padding: 2rem 1rem;
  `,
  cardName: styled.p`
    align-self: stretch;
    color: var(--Black, #35373a);
    font-family: 'Noto Sans KR';
    font-size: 1.1875rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,

  cardContainer: styled.div`
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    width: 100%;
    border-radius: 30px;
    overflow: hidden;
  `,

  checkContainer: styled.div`
    display: flex;
    width: 100%;
    padding: 2rem;
    flex-direction: column;
    align-items: flex-start;
    gap: 2rem;
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

export function MobileSettingPage({ cardId }: { cardId: number }) {
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

  return (
    <styles.pageContainer>
      <styles.cardName>
        {type === 'myCard' ? `마이 카드` : '메이트 카드'}
      </styles.cardName>
      <styles.cardContainer>
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
