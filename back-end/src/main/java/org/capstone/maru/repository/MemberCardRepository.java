package org.capstone.maru.repository;

import org.capstone.maru.domain.MemberCard;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface MemberCardRepository extends JpaRepository<MemberCard, Long> {

}
