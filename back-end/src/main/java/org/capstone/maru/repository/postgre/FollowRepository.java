package org.capstone.maru.repository.postgre;

import java.util.Optional;
import org.capstone.maru.domain.Follow;
import org.capstone.maru.domain.MemberAccount;
import org.springframework.data.jpa.repository.JpaRepository;

public interface FollowRepository extends JpaRepository<Follow, Long> {

    Optional<Follow> findByFollowerAndFollowing(MemberAccount followerAccount,
        MemberAccount followingAccount);
}
