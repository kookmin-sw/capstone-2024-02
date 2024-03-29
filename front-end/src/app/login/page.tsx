'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { Suspense, useEffect, useState } from 'react';

import { useAuthActions } from '@/features/auth';

function Page() {
  const router = useRouter();

  const searchParams = useSearchParams();
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState<string | null>(null);
  const [expiresIn, setExpiresIn] = useState<string | null>(null);

  const { login } = useAuthActions();

  useEffect(() => {
    setAccessToken(searchParams.get('access_token'));
    setRefreshToken(searchParams.get('refresh_token'));
    setExpiresIn(searchParams.get('expires_in'));
  }, [searchParams]);

  useEffect(() => {
    if (accessToken !== null && refreshToken !== null && expiresIn !== null) {
      login({ accessToken, refreshToken, expiresIn: +expiresIn });
      router.replace('/');
    }
  }, [accessToken, refreshToken, expiresIn, login, router]);

  return <></>;
}

export default function LoginPage() {
  return (
    <Suspense>
      <Page />
    </Suspense>
  );
}
