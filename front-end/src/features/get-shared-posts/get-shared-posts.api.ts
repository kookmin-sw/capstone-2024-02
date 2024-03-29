import axios from 'axios';

import { type GetSharedPostsDTO } from './get-shared-posts.dto';

export const getSharedPosts = async () => {
  const url = `/api/shared/posts`;
  return await axios.get<GetSharedPostsDTO>(encodeURI(url));
};
