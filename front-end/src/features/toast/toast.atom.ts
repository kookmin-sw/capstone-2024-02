import { atom } from 'recoil';

import { type ToastMessage } from './toast.type';

export const toastState = atom<Array<ToastMessage & { isVisible: boolean }>>({
  key: 'toastState',
  default: [],
});
