export interface SuccessBaseDTO {
  message: string;
  data: object | null | undefined;
}

export interface FailureDTO {
  status: number;
  code: string;
  errorMsg: string;
  reason: string;
}
