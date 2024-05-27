import { type SuccessBaseDTO } from '@/shared/types';

export interface GetUserDataDTO extends SuccessBaseDTO {
  data: {
    memberId: string;
    email: string;
    name: string;
    birthYear: string;
    gender: string;
    phoneNumber: string;
    initialized: boolean;
    myCardId: number;
    mateCardId: number;
    univCertified: boolean;
  };
}

export interface PostTokenRefreshDTO extends SuccessBaseDTO {
  data: {
    accessToken: string;
    refreshToken: string;
    expiresIn: number;
  };
}
