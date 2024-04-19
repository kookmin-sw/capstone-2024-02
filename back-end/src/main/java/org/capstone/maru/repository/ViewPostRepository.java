package org.capstone.maru.repository;

import org.capstone.maru.domain.ViewPost;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ViewPostRepository extends JpaRepository<ViewPost, Long> {

    Long countViewPostBySharedRoomPostId(Long sharedRoomPostId);
}
