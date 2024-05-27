'use client';

import { SettingPage } from '@/app/pages';
import { MobileSettingPage } from '@/app/pages/mobile';
import { useIsMobile } from '@/shared/mobile';

export default function Page({
  params: { cardId },
}: {
  params: { cardId: number };
}) {
  const isMobile = useIsMobile();
  return (
    <>
      {isMobile ? (
        <MobileSettingPage cardId={cardId} />
      ) : (
        <SettingPage cardId={cardId} />
      )}
    </>
  );
}
