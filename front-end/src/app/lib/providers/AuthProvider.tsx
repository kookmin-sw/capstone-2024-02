'use client';

import { usePathname, useRouter } from 'next/navigation';
import { useEffect } from 'react';

import {
  postTokenRefresh,
  useAuthActions,
  useAuthValue,
} from '@/features/auth';
import { load } from '@/shared/storage';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const auth = useAuthValue();
  const { login } = useAuthActions();

  const router = useRouter();
  const pathName = usePathname();

  useEffect(() => {
    if (pathName === '/login') {
      return;
    }

    if (auth === null) {
      const refreshToken = load<string>({ type: 'local', key: 'refreshToken' });
      if (refreshToken !== null) {
        postTokenRefresh(refreshToken)
          .then(({ data }) => {
            login({
              accessToken: data.accessToken,
              refreshToken,
              expiresIn: data.expiresIn,
            });
          })
          .catch(err => {
            console.error(err);
            router.replace('/');
          });
      } else {
        router.replace('/');
      }
    }
  }, [auth, login, router, pathName]);

  return <>{children}</>;
}
