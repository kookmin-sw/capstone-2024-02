import { atom } from 'recoil';

import { type Auth } from './auth.model';

export const authState = atom<Auth | null>({
  key: 'authState',
  default: null,
});
