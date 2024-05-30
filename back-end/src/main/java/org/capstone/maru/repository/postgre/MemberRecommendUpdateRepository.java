package org.capstone.maru.repository.postgre;

import org.capstone.maru.domain.MemberRecommendUpdate;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MemberRecommendUpdateRepository extends
    JpaRepository<MemberRecommendUpdate, String> {

}
