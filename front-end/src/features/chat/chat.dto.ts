import { type SuccessBaseDTO } from '@/shared/types';

export interface GetChatRoomDTO extends SuccessBaseDTO {
  data: [
    {
      roomId: number;
      roomName: string;
      unreadCount: number;
      lastMessage: string;
      lastMessageTime: string;
    },
  ];
}

export interface PostChatRoomDTO extends SuccessBaseDTO {
  data: {
    id: number;
  };
}

export interface GetChatRoomUserDTO extends SuccessBaseDTO {
  data: [
    {
      memberId: string;
      nickname: string;
      profileImageUrl: string;
    },
  ];
}

export interface PostChatRoomEnterDTO extends SuccessBaseDTO {
  data: [
    {
      messageId: string;
      sender: string;
      message: string;
      createdAt: string;
      nickname: string;
    },
  ];
}
