'use client';

import { ProfilePage } from '@/app/pages';
import { MobileProfilePage } from '@/app/pages/mobile';
import { useIsMobile } from '@/shared/mobile';

export default function Page({
  params: { memberId },
}: {
  params: { memberId: string };
}) {
  const isMobile = useIsMobile();
  return (
    <>
      {isMobile ? (
        <MobileProfilePage memberId={memberId} />
      ) : (
        <ProfilePage memberId={memberId} />
      )}
    </>
  );
}
