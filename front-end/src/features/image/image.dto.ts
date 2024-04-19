import { type SuccessBaseDTO } from '@/shared/types';

export interface GetImageURLDTO extends SuccessBaseDTO {
  data: {
    url: string;
    fileName: string;
  };
}
