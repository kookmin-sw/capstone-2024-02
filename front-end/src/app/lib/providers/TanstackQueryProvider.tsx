'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import React from 'react';

export function TanstackQueryProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [queryClient] = React.useState(() => new QueryClient({}));
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {/* 기본적으로 process.env.NODE_ENV === 'development'인 경우에만 번들에 포함되므로 프로덕션 빌드 중에 제외하는 것에 대해 걱정할 필요가 없음. */}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}
