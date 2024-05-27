'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

import { SearchBox } from './SearchBox';

import { useSearchUser, useUserProfile } from '@/features/profile';
import { useToast } from '@/features/toast';

export function UserSearchBox() {
  const router = useRouter();
  const [email, setEmail] = useState<string>('');

  const { mutate: search, data: searchUser } = useSearchUser(email);

  const { createToast } = useToast();

  const {
    mutate: mutateSearchUserProfile,
    data: searchUserProfile,
    error,
  } = useUserProfile(searchUser?.data.memberId ?? '');

  const handleEnter = () => {
    search();
  };

  useEffect(() => {
    if (searchUser?.data != null) {
      mutateSearchUserProfile();
    }
  }, [searchUser]);

  useEffect(() => {
    if (error != null) {
      createToast({
        message: '존재하지 않는 유저입니다.',
        option: { duration: 3000 },
      });
    }
    if (searchUserProfile != null)
      router.replace(`/profile/${searchUser?.data.memberId}`);
  }, [searchUserProfile, error]);
  return <SearchBox onContentChange={setEmail} onEnter={handleEnter} />;
}
