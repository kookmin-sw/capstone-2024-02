import { selector } from 'recoil';

import { authState } from './auth.atom';

export const authIsLogin = selector<boolean>({
  key: 'auth/isLogin',
  get: ({ get }) => {
    const auth = get(authState);
    return auth !== null;
  },
});
