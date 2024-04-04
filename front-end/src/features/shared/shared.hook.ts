import { useMutation, useQuery } from '@tanstack/react-query';
import { useCallback, useMemo, useState } from 'react';

import {
  deleteSharedPost,
  getSharedPost,
  getSharedPosts,
  scrapPost,
} from './shared.api';
import { type GetSharedPostsProps } from './shared.type';

import { type SuccessBaseDTO } from '@/shared/types';

export const usePaging = ({
  totalPages,
  sliceSize,
}: {
  totalPages: number;
  sliceSize: number;
}) => {
  const [page, setPage] = useState(1);

  const currentSlice = useMemo(
    () => Math.floor((page - 1) / sliceSize),
    [page, sliceSize],
  );
  const sliceCount = useMemo(
    () => Math.floor(totalPages / sliceSize),
    [totalPages, sliceSize],
  );

  const isFirstPage = useMemo(() => page === 1, [page]);
  const isLastPage = useMemo(() => page === totalPages, [page, totalPages]);

  const handleNextPage = useCallback(() => {
    if (isLastPage) return;
    setPage(prev => prev + 1);
  }, [isLastPage, setPage]);

  const handlePrevPage = useCallback(() => {
    if (isFirstPage) return;
    setPage(prev => prev - 1);
  }, [isFirstPage, setPage]);

  return useMemo(
    () => ({
      page,
      totalPages,
      sliceSize,
      currentSlice,
      sliceCount,
      isFirstPage,
      isLastPage,
      handleNextPage,
      handlePrevPage,
    }),
    [
      page,
      totalPages,
      sliceSize,
      currentSlice,
      sliceCount,
      isFirstPage,
      isLastPage,
      handleNextPage,
      handlePrevPage,
    ],
  );
};

export const useSharedPosts = ({
  filter,
  search,
  page,
  enabled,
}: GetSharedPostsProps & { enabled: boolean }) =>
  useQuery({
    queryKey: ['/api/shared/posts/studio', { filter, search, page }],
    queryFn: async () =>
      await getSharedPosts({ filter, search, page }).then(
        response => response.data,
      ),
    staleTime: 60000,
    enabled,
  });

export const useSharedPost = ({
  postId,
  enabled,
}: {
  postId: number;
  enabled: boolean;
}) =>
  useQuery({
    queryKey: [`/api/shared/posts/studio/${postId}`],
    queryFn: async () =>
      await getSharedPost(postId).then(response => response.data),
    enabled,
  });

export const useDeleteSharedPost = ({
  postId,
  onSuccess,
  onError,
}: {
  postId: number;
  onSuccess: (data: SuccessBaseDTO) => void;
  onError: (error: Error) => void;
}) =>
  useMutation({
    mutationFn: async () =>
      await deleteSharedPost(postId).then(response => response.data),
    onSuccess,
    onError,
  });

export const useScrapSharedPost = ({
  postId,
  onSuccess,
  onError,
}: {
  postId: number;
  onSuccess: (data: SuccessBaseDTO) => void;
  onError: (error: Error) => void;
}) =>
  useMutation({
    mutationFn: async () =>
      await scrapPost(postId).then(response => response.data),
    onSuccess,
    onError,
  });
