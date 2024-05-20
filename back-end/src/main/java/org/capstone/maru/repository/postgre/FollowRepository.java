package org.capstone.maru.repository.postgre;

import java.util.List;
import java.util.Optional;
import org.capstone.maru.domain.Follow;
import org.capstone.maru.domain.MemberAccount;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

public interface FollowRepository extends JpaRepository<Follow, Long> {

    Optional<Follow> findByFollowerAndFollowing(MemberAccount followerAccount,
        MemberAccount followingAccount);

    @Query("select fol.following.memberId from Follow fol where fol.follower.memberId = :followerId")
    List<String> findFollowingIdsByFollowerId(String followerId);
}
