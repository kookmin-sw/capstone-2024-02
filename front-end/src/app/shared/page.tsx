'use client';

import { SharedPostsPage } from '@/app/pages';
import { MobileSharedPostsPage } from '@/app/pages/mobile';
import { useIsMobile } from '@/shared/mobile';

export default function Page() {
  const isMobile = useIsMobile();
  return <>{isMobile ? <MobileSharedPostsPage /> : <SharedPostsPage />}</>;
}
