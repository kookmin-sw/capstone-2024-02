package org.capstone.maru.repository;

import java.util.Optional;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.repository.querydsl.StudioRoomPostCustomRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.EntityGraph.EntityGraphType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

@Repository
public interface StudioRoomPostRepository extends
    JpaRepository<StudioRoomPost, Long>,
    StudioRoomPostCustomRepository {

    Page<StudioRoomPost> findAllByPublisherGender(String gender, Pageable pageable);

    Optional<StudioRoomPost> findByIdAndPublisherGender(Long postId, String gender);

    void deleteByIdAndAndPublisherAccount_MemberId(Long postId, String memberId);
}
