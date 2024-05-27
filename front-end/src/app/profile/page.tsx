'use client';

import { UserInputPage } from '@/app/pages';
import { MobileUserInputPage } from '@/app/pages/mobile';
import { useIsMobile } from '@/shared/mobile';

export default function Page() {
  const isMobile = useIsMobile();
  return <>{isMobile ? <MobileUserInputPage /> : <UserInputPage />}</>;
}
