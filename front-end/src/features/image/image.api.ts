import axios from 'axios';

import { type GetImageURLDTO } from './image.dto';

export const getImageURL = async (fileExtension: string) =>
  await axios.get<GetImageURLDTO>(
    `/maru-api/image/upload?extension=${fileExtension}`,
  );

export const putImage = async (url: string, image: File) =>
  await axios.put(url, image, {
    headers: {
      'Content-Type': image.type,
      Authorization: null,
    },
  });
