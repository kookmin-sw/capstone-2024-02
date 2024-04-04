export type StorageType = 'session' | 'local';

export const save = <T>({
  type,
  key,
  value,
}: {
  type: StorageType;
  key: string;
  value: T;
}) => {
  if (typeof window === 'undefined') return;

  const storage = type === 'local' ? localStorage : sessionStorage;
  storage.setItem(key, JSON.stringify(value));
};

export const load = <T>({ type, key }: { type: StorageType; key: string }) => {
  if (typeof window === 'undefined') return null;

  const storage = type === 'local' ? localStorage : sessionStorage;
  const storedValue = storage.getItem(key);
  if (storedValue === null) return null;

  const parsedValue = JSON.parse(storedValue) as T;
  return parsedValue;
};

export const remove = ({ type, key }: { type: StorageType; key: string }) => {
  if (typeof window === 'undefined') return;

  const storage = type === 'local' ? localStorage : sessionStorage;
  storage.removeItem(key);
};
