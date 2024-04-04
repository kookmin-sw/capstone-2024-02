import { type AtomEffect, type DefaultValue } from 'recoil';

import { load, remove, save, type StorageType } from '@/shared/storage';

export const storageEffect =
  <StoredType>({
    key,
    storageType,
  }: {
    key: string;
    storageType: StorageType;
  }): AtomEffect<StoredType> =>
  ({ setSelf, onSet }) => {
    if (typeof window === 'undefined') return;

    const savedValue = load<StoredType>({
      type: storageType,
      key,
    });
    if (savedValue !== null) {
      setSelf(savedValue);
    }

    onSet((newValue: StoredType | DefaultValue, _, isReset: boolean) => {
      if (isReset) {
        remove({ type: storageType, key });
      } else {
        save({ type: storageType, key, value: newValue });
      }
    });
  };
