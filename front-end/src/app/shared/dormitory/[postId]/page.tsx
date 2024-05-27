'use client';

import { SharedPostPage } from '@/app/pages';
import { MobileSharedPostPage } from '@/app/pages/mobile';
import { useIsMobile } from '@/shared/mobile';

export default function Page({
  params: { postId },
}: {
  params: { postId: string };
}) {
  const isMobile = useIsMobile();
  return (
    <>
      {isMobile ? (
        <MobileSharedPostPage postId={+postId} type="dormitory" />
      ) : (
        <SharedPostPage postId={+postId} type="dormitory" />
      )}
    </>
  );
}
