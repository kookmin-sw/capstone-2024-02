import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import React from 'react';
import './globals.scss';

import {
  StyledComponentsRegistry,
  TanstackQueryProvider,
  AuthProvider,
} from '@/app/lib/providers';
import { NavigationBar, FloatingChatting } from '@/components';

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
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" />
        <link
          href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@400..800&display=swap"
          rel="stylesheet"
        />
        <link
          rel="stylesheet"
          as="style"
          href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@100..900&display=swap"
          rel="stylesheet"
        />
        <script
          async
          type="text/javascript"
          src={`https://oapi.map.naver.com/openapi/v3/maps.js?ncpClientId=${process.env.NEXT_PUBLIC_NAVER_MAP_CLIENT_ID}`}
        />
      </head>
      <body className={inter.className}>
        <TanstackQueryProvider>
          <StyledComponentsRegistry>
            <AuthProvider>
              <NavigationBar />
              <main>{children}</main>
              <FloatingChatting />
            </AuthProvider>
          </StyledComponentsRegistry>
        </TanstackQueryProvider>
      </body>
    </html>
  );
}
