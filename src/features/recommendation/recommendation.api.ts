import axios from 'axios';

import { type GetRecommendationMateDTO } from './recommendation.dto';

export const getRecommendationMate = async (
  memberId: string,
  cardType: 'my' | 'mate',
) =>
  await axios.get<GetRecommendationMateDTO>(
    `http://localhost:8000/recommendation/${memberId}/${cardType}`,
  );
