type storeType = 'session' | 'local';

export const save = ({
  type,
  key,
  value,
}: {
  type: storeType;
  key: string;
  value: string;
}) => {
  if (type === 'local') {
    localStorage.setItem(key, value);
  } else {
    sessionStorage.setItem(key, value);
  }
};

export const load = ({ type, key }: { type: storeType; key: string }) => {
  if (type === 'local') {
    return localStorage.getItem(key);
  }
  return sessionStorage.getItem(key);
};

export const remove = ({ type, key }: { type: storeType; key: string }) => {
  if (type === 'local') {
    localStorage.removeItem(key);
  } else {
    sessionStorage.removeItem(key);
  }
};
