'use client';

import { LandingPage, MainPage } from './pages';
import { MobileLandingPage, MobileMainPage } from './pages/mobile';

import { useAuthIsLogin } from '@/features/auth';
import { useIsMobile } from '@/shared/mobile';

export default function Home() {
  const isLogin = useAuthIsLogin();
  const isMobile = useIsMobile();
  return (
    <>
      {isLogin ? (
        <>{isMobile ? <MobileMainPage /> : <MainPage />}</>
      ) : (
        <>{isMobile ? <MobileLandingPage /> : <LandingPage />}</>
      )}
    </>
  );
}
