import axios from 'axios';

import { type PostTokenRefreshDTO, type GetUserDataDTO } from './auth.dto';

export const getUserData = async () =>
  await axios.get<GetUserDataDTO>('/api/auth/initial/info');

export const postTokenRefresh = async (refreshToken: string) =>
  await axios.post<PostTokenRefreshDTO>(
    '/api/auth/token/refresh',
    {},
    { headers: { Authorization: `Bearer ${refreshToken}` } },
  );

export const getAuthLogout = async (refreshToken: string) =>
  await axios.get('/api/auth/logout', {
    headers: { Authorization: `Bearer ${refreshToken}` },
  });
