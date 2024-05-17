import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import React from 'react';
import './globals.scss';
import './lib/axios';

import {
  AuthProvider,
  RecoilRootProvider,
  StyledComponentsRegistry,
  TanstackQueryProvider,
  ToastProvider,
} from '@/app/lib/providers';
import { FloatingChatting, NavigationBar } from '@/components';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Maru',
  description: 'Generated by create next app',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <head>
        <script
          async
          type="text/javascript"
          src={`https://oapi.map.naver.com/openapi/v3/maps.js?ncpClientId=${process.env.NEXT_PUBLIC_NAVER_MAP_CLIENT_ID}&submodules=geocoder`}
        />
      </head>
      <body className={inter.className}>
        <TanstackQueryProvider>
          <RecoilRootProvider>
            <StyledComponentsRegistry>
              <NavigationBar />
              <AuthProvider>
                <main>
                  {children}
                  <FloatingChatting />
                </main>
                <ToastProvider />
              </AuthProvider>
            </StyledComponentsRegistry>
          </RecoilRootProvider>
        </TanstackQueryProvider>
      </body>
    </html>
  );
}
