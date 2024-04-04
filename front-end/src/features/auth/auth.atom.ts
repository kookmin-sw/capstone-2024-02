import { atom } from 'recoil';

import { type Auth } from './auth.type';

export const authState = atom<Auth | null>({
  key: 'authState',
  default: null,
});
