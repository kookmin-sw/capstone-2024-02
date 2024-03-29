'use client';

import { usePathname, useRouter } from 'next/navigation';
import { useEffect } from 'react';

import {
  postTokenRefresh,
  useAuthActions,
  useAuthState,
} from '@/features/auth';
import { load } from '@/shared/persist';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [auth] = useAuthState();
  const { login } = useAuthActions();

  const router = useRouter();
  const pathName = usePathname();

  useEffect(() => {
    if (pathName === '/login') {
      return;
    }

    if (auth === null) {
      const refreshToken = load({ type: 'local', key: 'refreshToken' });
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
