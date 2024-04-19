import { useCallback, useMemo } from 'react';
import { useRecoilState } from 'recoil';

import { toastState } from './toast.atom';
import { type ToastMessage } from './toast.type';

export const useToast = () => {
  const [toast, setToast] = useRecoilState(toastState);

  const createToast = useCallback(
    ({
      message,
      option,
    }: {
      message: ToastMessage['message'];
      option: ToastMessage['option'];
    }) => {
      const id = crypto.randomUUID();
      setToast(prev => prev.concat({ id, message, option, isVisible: true }));
      setTimeout(() => {
        setToast(prev =>
          prev.map(value =>
            value.id === id ? { ...value, isVisible: false } : value,
          ),
        );
        setTimeout(() => {
          setToast(prev => prev.filter(value => value.id !== id));
        }, 200);
      }, option.duration);
    },
    [setToast],
  );

  return useMemo(() => ({ toast, createToast }), [toast, createToast]);
};
