package org.capstone.maru.repository.postgre;

import java.util.List;
import org.capstone.maru.domain.SharedRoomPost;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

public interface SharedRoomPostRepository extends JpaRepository<SharedRoomPost, Long> {

    @Query("select srp from SharedRoomPost srp where srp.publisherAccount.memberId =:memberId")
    List<SharedRoomPost> findAllByPublisherAccount_MemberId(String memberId);
}
