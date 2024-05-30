'use client';

import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

import { useAuthActions, useAuthValue, useUserData } from '@/features/auth';

export function InitialzationProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();

  const auth = useAuthValue();
  const { setAuthUserData } = useAuthActions();

  const { data: userData } = useUserData(auth?.accessToken != null);

  useEffect(() => {
    if (userData != null) {
      setAuthUserData(userData);
      if (userData.initialized) {
        router.replace('/profile');
      }
    }
  }, [userData, router, setAuthUserData]);

  return <>{children}</>;
}
