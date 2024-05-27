import { atom } from 'recoil';

export const chatOpenState = atom<boolean>({
  key: 'isChatOpenState',
  default: false,
});
