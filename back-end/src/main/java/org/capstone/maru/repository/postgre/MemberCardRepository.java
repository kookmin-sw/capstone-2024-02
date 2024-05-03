package org.capstone.maru.repository.postgre;

import org.capstone.maru.domain.FeatureCard;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface MemberCardRepository extends JpaRepository<FeatureCard, Long> {

}
