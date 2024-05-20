package org.capstone.maru.repository.postgre;

import java.util.List;
import org.capstone.maru.domain.Recommend;
import org.springframework.data.jpa.repository.JpaRepository;

public interface RecommendRepository extends JpaRepository<Recommend, Long> {

    List<Recommend> findAllByUserIdAndCardTypeAndRecommendationCardTypeOrderByScoreDesc(
        String userId, String cardType, String recommendationCardType);

}
