import { type User } from '@/entities/user';

export interface Auth {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  user?: User;
}
