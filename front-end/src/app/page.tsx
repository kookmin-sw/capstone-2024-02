'use client';

import { LandingPage, MainPage } from './pages';

import { useAuthIsLogin } from '@/features/auth';

export default function Home() {
  const isLogin = useAuthIsLogin();
  return <>{isLogin ? <MainPage /> : <LandingPage />}</>;
}
