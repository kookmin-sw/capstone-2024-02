'use client';

import { WritingPostPage } from '@/app/pages';
import { MobileWritingPostPage } from '@/app/pages/mobile';
import { useIsMobile } from '@/shared/mobile';

export default function Page() {
  const isMobile = useIsMobile();
  return (
    <>
      {isMobile ? (
        <MobileWritingPostPage type="hasRoom" />
      ) : (
        <WritingPostPage type="hasRoom" />
      )}
    </>
  );
}
