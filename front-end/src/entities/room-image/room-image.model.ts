import { type StudioRoomPost } from '../studio-room-post';

export interface RoomImage {
  id: number;
  fileName: string;
  storeImagePath: string;
  isThumbnail: boolean;
  studioRoomPost: StudioRoomPost;
}
