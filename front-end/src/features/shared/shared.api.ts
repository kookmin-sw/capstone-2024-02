import axios from 'axios';

import {
  type GetDormitorySharedPostDTO,
  type GetDormitorySharedPostsDTO,
  type GetSharedPostDTO,
  type GetSharedPostsDTO,
} from './shared.dto';
import { type GetSharedPostsProps, type SharedPostProps } from './shared.type';

import { type SuccessBaseDTO } from '@/shared/types';

export const getSharedPosts = async ({
  filter,
  cardOption,
  search,
  page,
}: GetSharedPostsProps) => {
  const getURI = () => {
    const baseURL = '/maru-api/shared/posts/studio';
    let query = '';

    if (
      filter != null &&
      Object.keys(filter).length > 0 &&
      JSON.stringify(filter).length > 2
    ) {
      query += `filter=${JSON.stringify(filter)}`;
    }

    if (search != null) {
      query += `&search=${search}`;
    }

    query += `&page=${page}`;

    return `${baseURL}?${encodeURIComponent(query)}&cardOption=${cardOption}`;
  };

  return await axios.get<GetSharedPostsDTO>(getURI());
};

export const createSharedPost = async (postData: SharedPostProps) =>
  await axios.post<SuccessBaseDTO>(`/maru-api/shared/posts/studio`, postData);

export const updateSharedPost = async ({
  postId,
  postData,
}: {
  postId: number;
  postData: SharedPostProps;
}) =>
  await axios.put<SuccessBaseDTO>(
    `/maru-api/shared/posts/studio/${postId}`,
    postData,
  );

export const getSharedPost = async (postId: number) =>
  await axios.get<GetSharedPostDTO>(`/maru-api/shared/posts/studio/${postId}`);

export const deleteSharedPost = async (postId: number) =>
  await axios.delete<SuccessBaseDTO>(`/maru-api/shared/posts/studio/${postId}`);

export const scrapPost = async (postId: number) =>
  await axios.post<SuccessBaseDTO>(
    `/maru-api/shared/posts/studio/${postId}/scrap`,
  );

export const getDormitorySharedPosts = async ({
  filter,
  cardOption,
  search,
  page,
}: GetSharedPostsProps) => {
  const getURI = () => {
    const baseURL = '/maru-api/shared/posts/dormitory';
    let query = '';

    if (
      filter != null &&
      Object.keys(filter).length > 0 &&
      JSON.stringify(filter).length > 2
    ) {
      query += `filter=${JSON.stringify(filter)}`;
    }

    if (search != null) {
      query += `&search=${search}`;
    }

    query += `&page=${page}`;

    return `${baseURL}?${encodeURIComponent(query)}&cardOption=${cardOption}`;
  };

  return await axios.get<GetDormitorySharedPostsDTO>(getURI());
};

export const createDormitorySharedPost = async (postData: SharedPostProps) =>
  await axios.post<SuccessBaseDTO>(
    `/maru-api/shared/posts/dormitory`,
    postData,
  );

export const getDormitorySharedPost = async (postId: number) =>
  await axios.get<GetDormitorySharedPostDTO>(
    `/maru-api/shared/posts/dormitory/${postId}`,
  );

export const updateDormitorySharedPost = async ({
  postId,
  postData,
}: {
  postId: number;
  postData: SharedPostProps;
}) =>
  await axios.put<SuccessBaseDTO>(
    `/maru-api/shared/posts/dormitory/${postId}`,
    postData,
  );

export const deleteDormitorySharedPost = async (postId: number) =>
  await axios.delete<SuccessBaseDTO>(
    `/maru-api/shared/posts/dormitory/${postId}`,
  );

export const scrapDormitoryPost = async (postId: number) =>
  await axios.post<SuccessBaseDTO>(
    `/maru-api/shared/posts/dormitory/${postId}/scrap`,
  );
