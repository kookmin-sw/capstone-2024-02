package org.capstone.maru.repository.postgre;

import java.util.Optional;
import org.capstone.maru.domain.DormitoryRoomPost;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.repository.postgre.querydsl.DormitoryRoomPostCustomRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DormitoryRoomPostRepository extends
    JpaRepository<DormitoryRoomPost, Long>,
    DormitoryRoomPostCustomRepository {

    Page<DormitoryRoomPost> findAllByPublisherGender(String gender, Pageable pageable);

    Optional<DormitoryRoomPost> findByIdAndPublisherGender(Long postId, String gender);

    Optional<DormitoryRoomPost> findByIdAndPublisherAccount_MemberId(Long postId, String memberId);

    void deleteByIdAndPublisherAccount_MemberId(Long postId, String memberId);
}
