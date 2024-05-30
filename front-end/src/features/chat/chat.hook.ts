import { useMutation, useQuery } from '@tanstack/react-query';

import {
  getChatRoomList,
  getChatRoomUser,
  postChatRoom,
  getEnterChatRoom,
  postInviteUser,
  postExitChatRoom,
  putChatRoomName,
  deleteChatRoom,
} from './chat.api';

export const useChatRoomList = (token: string | undefined) =>
  useQuery({
    queryKey: [`/chatRoom`, token],
    queryFn: async () => await getChatRoomList(),
    enabled: token !== undefined,
  });

export const useCreateChatRoom = () =>
  useMutation({
    mutationFn: postChatRoom,
  });

export const useInviteUsers = (roomId: number, members: string[]) =>
  useMutation({
    mutationFn: async () => {
      await postInviteUser(roomId, members);
    },
  });

export const useEnterChatRoom = (roomId: number, page: number, size: number) =>
  useQuery({
    queryKey: [`/chatRoom/${roomId}/chat`, page, size],
    queryFn: async () => await getEnterChatRoom(roomId, page, size),
  });

export const useExitChatRoom = (roomId: number) =>
  useMutation({
    mutationFn: async () => {
      await postExitChatRoom(roomId);
    },
  });

export const useChatRoomUser = (roomId: number) =>
  useQuery({
    queryKey: [`/chatRoom/${roomId}`],
    queryFn: async () => await getChatRoomUser(roomId),
    enabled: roomId !== undefined,
  });

export const useChangeChatRoomName = (roomName: string, roomId: number) =>
  useMutation({
    mutationFn: async () => {
      await putChatRoomName(roomName, roomId);
    },
  });

export const useDeleteChatRoom = () =>
  useMutation({
    mutationFn: deleteChatRoom,
  });
