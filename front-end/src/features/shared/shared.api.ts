import axios from 'axios';

import { type GetSharedPostDTO, type GetSharedPostsDTO } from './shared.dto';
import {
  type GetSharedPostsProps,
  type GetSharedPostsFilter,
} from './shared.type';

import {
  RentalTypeValue,
  RoomTypeValue,
  type SuccessBaseDTO,
} from '@/shared/types';

const filterConvertToValues = (filter: GetSharedPostsFilter) => {
  const result: Partial<Record<keyof GetSharedPostsFilter, number[]>> = {};

  if (filter.roomType !== undefined) {
    result.roomType = Object.values(filter.roomType).map(
      value => RoomTypeValue[value],
    );
  }

  if (filter.rentalType !== undefined) {
    result.rentalType = Object.values(filter.rentalType).map(
      value => RentalTypeValue[value],
    );
  }

  return result;
};

export const getSharedPosts = async ({
  filter,
  search,
  page,
}: GetSharedPostsProps) => {
  const getURI = () => {
    const baseURL = '/api/shared/posts/studio';
    let query = '';

    if (filter !== undefined) {
      query += `filter=${JSON.stringify(filterConvertToValues(filter))}`;
    }

    if (search !== undefined) {
      query += `&search=${search}`;
    }

    if (page !== undefined) {
      query += `&page=${page}`;
    }

    if (query === '') return baseURL;
    return `${baseURL}?${encodeURIComponent(query)}`;
  };

  return await axios.get<GetSharedPostsDTO>(getURI());
};

export const getSharedPost = async (postId: number) =>
  await axios.get<GetSharedPostDTO>(`/api/shared/posts/studio/${postId}`);

export const deleteSharedPost = async (postId: number) =>
  await axios.delete<SuccessBaseDTO>(`/api/shared/posts/studio/${postId}`);

export const scrapPost = async (postId: number) =>
  await axios.get<SuccessBaseDTO>(`/api/shared/posts/studio/${postId}/scrap`);
