export interface GetUserDataDTO {
  memberId: string;
  email: string;
  name: string;
  birthYear: string;
  gender: string;
  phoneNumber: string;
  initialized: boolean;
}

export interface PostTokenRefreshDTO {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}
