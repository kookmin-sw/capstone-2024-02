import axios from 'axios';

import {
  type PostChatRoomEnterDTO,
  type GetChatRoomDTO,
  type GetChatRoomUserDTO,
  type PostChatRoomDTO,
} from './chat.dto';

export const getChatRoomList = async () =>
  await axios.get<GetChatRoomDTO>(`/maru-api/chatRoom`).then(res => res.data);

export const postChatRoom = async ({
  roomName,
  members,
  myID,
}: {
  roomName: string;
  members: string[];
  myID: string;
}) => {
  const list = await getChatRoomList();

  const diffArray = (arr1: string[], arr2: string[]) => {
    if (arr1.length !== 2) return false;
    for (let i = 0; i < 2; i += 1) {
      if (arr1[i] !== arr2[i]) return false;
    }
    return true;
  };

  const createFlag = async () => {
    const flagPromises = list.data.map(async room => {
      const users = await getChatRoomUser(room.roomId);
      const userIDArr = users.data.map(user => user.memberId);
      const newArray = [...members, myID];
      const sortedArray1 = userIDArr.slice().sort();
      const sortedArray2 = newArray.slice().sort();

      return diffArray(sortedArray1, sortedArray2);
    });

    const flags = await Promise.all(flagPromises);

    if (flags.includes(true)) {
      return false;
    }

    return true;
  };

  createFlag().then(flag => {
    if (flag) {
      axios
        .post<PostChatRoomDTO>(`/maru-api/chatRoom`, {
          roomName,
          members,
        })
        .then(res => res.data);
    }
  });
};

export const postInviteUser = async (roomId: number, members: string[]) => {
  await axios
    .post(`/maru-api/chatRoom/${roomId}/invite`, {
      members,
    })
    .then(res => res.data);
};

export const getEnterChatRoom = async (
  roomId: number,
  page: number,
  size: number,
) => {
  const res = await axios.get<PostChatRoomEnterDTO>(
    `/maru-api/chatRoom/${roomId}/chat`,
    {
      params: {
        roomId,
        page,
        size,
      },
    },
  );
  console.log(res.data);
  return res.data;
};

export const postExitChatRoom = async (roomId: number) => {
  await axios.post(`/maru-api/chatRoom/${roomId}/exit`).then(res => res.data);
};

export const getChatRoomUser = async (roomId: number) =>
  await axios
    .get<GetChatRoomUserDTO>(`/maru-api/chatRoom/${roomId}`)
    .then(res => res.data);

export const putChatRoomName = async (roomName: string, roomId: number) =>
  await axios
    .patch(`/maru-api/chatRoom/${roomId}`, {
      roomName: roomName,
    })
    .then(res => res.data);

export const deleteChatRoom = async (roomId: number) =>
  await axios.delete(`/maru-api/chatRoom/${roomId}/exit`);
