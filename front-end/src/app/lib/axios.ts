'use client';

import axios, { type AxiosError, type AxiosResponse } from 'axios';

import { postTokenRefresh } from '@/features/auth';
import { load, save } from '@/shared/storage';
import { type FailureDTO } from '@/shared/types';

let isRefreshing = false;
let refreshSubscribers: Array<(token: string) => void> = [];

const subscribeTokenRefresh = (callback: (token: string) => void) => {
  refreshSubscribers.push(callback);
};

const onRefreshed = (token: string) => {
  refreshSubscribers.forEach(callback => {
    callback(token);
  });
  refreshSubscribers = [];
};

axios.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError<FailureDTO>) => {
    if (error.response?.status !== 401 || error.response?.data.code !== 'C002')
      return await Promise.reject(error);

    const refreshToken = load<string>({ type: 'local', key: 'refreshToken' });
    if (refreshToken == null) {
      window.location.href = '/';
      return await Promise.reject(error);
    }

    if (isRefreshing) {
      return await new Promise(resolve => {
        subscribeTokenRefresh((token: string) => {
          const { config } = error;
          if (config?.headers != null) {
            config.headers.Authorization = `Bearer ${token}`;
            resolve(axios(config));
          }
        });
      });
    }

    isRefreshing = true;
    try {
      const {
        data: { accessToken, refreshToken: newRefreshToken, expiresIn },
      } = await postTokenRefresh(refreshToken);

      axios.defaults.headers.common.Authorization = `Bearer ${accessToken}`;
      save({ type: 'local', key: 'refreshToken', value: newRefreshToken });
      save({ type: 'local', key: 'expiresIn', value: `${expiresIn}` });

      const { config } = error;
      onRefreshed(accessToken);
      if (config != null) {
        config.headers.Authorization = `Bearer ${accessToken}`;
        return await axios(config);
      }
    } catch (refreshError) {
      window.location.href = '/';
      return await Promise.reject(error);
    } finally {
      isRefreshing = false;
    }

    return await Promise.reject(error);
  },
);
