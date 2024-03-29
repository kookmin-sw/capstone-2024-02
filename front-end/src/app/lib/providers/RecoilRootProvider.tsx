'use client';

import { RecoilRoot } from 'recoil';

export function RecoilRootProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  return <RecoilRoot>{children}</RecoilRoot>;
}
