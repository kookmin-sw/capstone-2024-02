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

    query += `page=${page}`;

    if (filter != null) {
      query += `&filter=${JSON.stringify(filter)}`;
    }

    if (search != null) {
      query += `&search=${search}`;
    }

    return `${baseURL}?${encodeURI(query)}&cardOption=${cardOption}`;
  };

  return await axios.get<GetSharedPostsDTO>(getURI()).then(res => ({
    totalPages: res.data.data.totalPages,
    data: res.data.data.content,
  }));
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

    query += `page=${page}`;

    if (filter != null) {
      query += `&filter=${JSON.stringify(filter)}`;
    }

    if (search != null) {
      query += `&search=${search}`;
    }

    return `${baseURL}?${encodeURI(query)}&cardOption=${cardOption}`;
  };

  return await axios.get<GetDormitorySharedPostsDTO>(getURI()).then(res => ({
    totalPages: res.data.data.totalPages,
    data: res.data.data.content,
  }));
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
