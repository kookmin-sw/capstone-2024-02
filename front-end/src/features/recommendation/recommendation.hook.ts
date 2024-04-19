import { useQuery } from '@tanstack/react-query';

import { getRecommendationMate } from './recommendation.api';

import { type CardType } from '@/entities/shared-posts-filter';

export const useRecommendationMate = ({
  memberId,
  cardType,
  enabled,
}: {
  memberId: string;
  cardType: CardType;
  enabled: boolean;
}) =>
  useQuery({
    queryKey: ['/api/match/recommendation', { memberId, cardType }],
    queryFn: async () =>
      await getRecommendationMate(memberId, cardType).then(
        response => response.data.recommendation,
      ),
    enabled,
  });
