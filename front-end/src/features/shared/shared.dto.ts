import {
  type SharedPost,
  type SharedPostListItem,
} from '@/entities/shared-post';
import { type SuccessBaseDTO } from '@/shared/types';

export interface GetSharedPostsDTO extends SuccessBaseDTO {
  message: string;
  data: {
    content: SharedPostListItem[];
    pageable: {
      pageNumber: number;
      pageSize: number;
      sort: {
        empty: boolean;
        unsorted: boolean;
        sorted: boolean;
      };
      offset: number;
      paged: boolean;
      unpaged: boolean;
    };
    last: boolean;
    totalPages: number;
    totalElements: number;
    first: boolean;
    size: number;
    number: number;
    sort: {
      empty: boolean;
      unsorted: boolean;
      sorted: boolean;
    };
    numberOfElements: number;
    empty: boolean;
  };
}

export interface GetSharedPostDTO extends SuccessBaseDTO {
  data: SharedPost;
}
