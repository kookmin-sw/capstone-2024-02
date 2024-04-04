import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { useCallback, useMemo } from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';

import { getUserData } from './auth.api';
import { authState } from './auth.atom';
import { authIsLogin } from './auth.selector';
import { type Auth } from './auth.type';

import { type User } from '@/entities/user';
import { remove, save } from '@/shared/storage';

export const useAuthValue = () => useRecoilValue(authState);
export const useAuthIsLogin = () => useRecoilValue(authIsLogin);

export const useAuthActions = () => {
  const [, setAuth] = useRecoilState(authState);

  const login = useCallback(
    (auth: Auth) => {
      axios.defaults.headers.common.Authorization = `Bearer ${auth.accessToken}`;
      save({ type: 'local', key: 'refreshToken', value: auth.refreshToken });
      save({ type: 'local', key: 'expiresIn', value: `${auth.expiresIn}` });
      setAuth(auth);
    },
    [setAuth],
  );

  const logout = useCallback(() => {
    axios.defaults.headers.common.Authorization = '';
    remove({ type: 'local', key: 'refreshToken' });
    remove({ type: 'local', key: 'expiresIn' });
    setAuth(null);
  }, [setAuth]);

  const setAuthUserData = useCallback(
    (user: User) => {
      setAuth(prev => {
        if (prev === null) return null;
        return {
          ...prev,
          user,
        };
      });
    },
    [setAuth],
  );

  return useMemo(
    () => ({ login, logout, setAuthUserData }),
    [login, logout, setAuthUserData],
  );
};

export const useUserData = (enabled: boolean) =>
  useQuery({
    queryKey: ['/api/auth/initial/info'],
    queryFn: async () => await getUserData().then(response => response.data),
    enabled,
  });
