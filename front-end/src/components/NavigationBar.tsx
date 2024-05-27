'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import styled from 'styled-components';

import { UserSearchBox } from './UserSearchBox';

import {
  getAuthLogout,
  useAuthActions,
  useAuthIsLogin,
  useAuthValue,
  useUserData,
} from '@/features/auth';
import { useToast } from '@/features/toast';
import { useIsMobile } from '@/shared/mobile';
import { load } from '@/shared/storage';

const styles = {
  container: styled.nav`
    display: flex;
    flex: 1;
    min-width: 1440px;
    width: 100%;
    height: 4.5rem;
    padding: 1rem 11.25rem;
    align-items: center;
    justify-content: space-between;

    border-bottom: 1px solid #f7f6f9;
    background: #fff;
    box-shadow: 0px 0px 20px -2px rgba(0, 0, 0, 0.05);
    z-index: 2147483647;

    @media (max-width: 768px) {
      min-width: 390px;
      width: 100vw;
      padding: 1rem 2rem;
      z-index: 10000;
    }
  `,
  mobileMenuIcon: styled.div`
    display: none;

    @media (max-width: 768px) {
      display: block;
      width: 1rem;
      height: 1rem;
      flex-shrink: 0;
      background: url('/kebab-horizontal.svg') no-repeat;
      cursor: pointer;
    }
  `,
  menuContainer: styled.ul`
    display: none;

    @media (max-width: 768px) {
      display: flex;
      flex-direction: column;
      width: 15rem;
      padding: 1rem;
      background: #fff;
      z-index: 20000;
      box-shadow: 0px 4px 20px 0px rgba(0, 0, 0, 0.2);
      border-radius: 1.25rem;
      position: fixed;
      top: 5rem;
      right: 0.5rem;
    }
  `,
  menuItem: styled.div`
    display: flex;
    padding: 0.8rem;
    align-items: flex-start;
    gap: 0.8rem;
    align-self: stretch;
    border-bottom: 1px solid #e5e5ea;
    cursor: pointer;

    p {
      color: #2c2c2e;
      font-family: 'Noto Sans KR';
      font-size: 0.875rem;
      font-style: normal;
      font-weight: 400;
      line-height: normal;
    }
  `,
  utils: styled.div`
    display: flex;
    align-items: center;
    gap: 3.63rem;
    @media (max-width: 768px) {
      display: none;
    }
  `,
  title: styled.h1`
    color: var(--Main-1, #e15637);
    font-family: 'Baloo 2';
    font-size: 1.875rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
  `,
  links: styled.div`
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1.5rem;
    @media (max-width: 768px) {
      display: none;
    }
  `,
  logout: styled.button`
    all: unset;
    display: flex;
    padding: 0.5rem 1rem;
    align-items: flex-start;

    border-radius: 8px;
    background: #e15637;

    color: #fff;
    font-family: 'Noto Sans KR';
    font-size: 1rem;
    font-style: normal;
    font-weight: 700;
    line-height: normal;

    cursor: pointer;
    @media (max-width: 768px) {
      display: none;
    }
  `,
};

export function NavigationBar() {
  const isLogin = useAuthIsLogin();
  const router = useRouter();

  const isMobile = useIsMobile();

  const auth = useAuthValue();
  const { logout } = useAuthActions();
  const { createToast } = useToast();

  const { data } = useUserData(auth?.accessToken !== undefined);

  const handleLogout = () => {
    const refreshToken = load<string>({ type: 'local', key: 'refreshToken' });
    router.replace('/');
    if (refreshToken != null) {
      getAuthLogout(refreshToken)
        .then(() => {
          logout();
          createToast({
            message: '로그아웃이 정상적으로 이루어졌습니다.',
            option: {
              duration: 3000,
            },
          });
        })
        .catch(() => {
          createToast({
            message: '로그아웃에 실패했습니다.',
            option: {
              duration: 3000,
            },
          });
        });
    }
  };

  const [isMenuClick, setIsMenuClick] = useState(false);

  return (
    <>
      <styles.container>
        {isMobile ? (
          <>
            <styles.title
              onClick={() => {
                setIsMenuClick(false);
              }}
            >
              <Link href="/">maru</Link>
            </styles.title>
            <styles.mobileMenuIcon
              onClick={() => {
                setIsMenuClick(prev => !prev);
              }}
            />
          </>
        ) : null}
        <styles.utils>
          <styles.title>
            <Link href="/">maru</Link>
          </styles.title>
          <UserSearchBox />
        </styles.utils>
        <styles.links>
          <Link href="/shared">메이트찾기</Link>
          <Link href={`/profile/${data?.memberId}`}>마이페이지</Link>
          {isLogin && (
            <styles.logout
              onClick={() => {
                handleLogout();
              }}
            >
              로그아웃
            </styles.logout>
          )}
        </styles.links>
      </styles.container>
      {isMenuClick && (
        <styles.menuContainer>
          <Link href="/shared">
            <styles.menuItem
              onClick={() => {
                setIsMenuClick(false);
              }}
            >
              <p>메이트 찾기</p>
            </styles.menuItem>
          </Link>
          <Link href={`/profile/${data?.memberId}`}>
            <styles.menuItem
              onClick={() => {
                setIsMenuClick(false);
              }}
              style={{
                borderBottom: !isLogin ? 'none' : '1px solid #e5e5ea',
              }}
            >
              <p>마이페이지</p>
            </styles.menuItem>
          </Link>
          {isLogin && (
            <styles.menuItem
              onClick={() => {
                setIsMenuClick(false);
                handleLogout();
              }}
              style={{
                borderBottom: 'none',
              }}
            >
              <p>로그아웃</p>
            </styles.menuItem>
          )}
        </styles.menuContainer>
      )}
    </>
  );
}
