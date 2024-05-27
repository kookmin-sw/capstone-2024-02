'use client';

import { WritingPostPage } from '@/app/pages';
import { MobileWritingPostPage } from '@/app/pages/mobile';
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
        <MobileWritingPostPage type="dormitory" postId={+postId} />
      ) : (
        <WritingPostPage type="dormitory" postId={+postId} />
      )}
    </>
  );
}
