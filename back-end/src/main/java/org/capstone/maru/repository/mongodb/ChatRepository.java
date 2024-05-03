package org.capstone.maru.repository.mongodb;

import java.time.LocalDateTime;
import java.util.List;
import org.capstone.maru.domain.Chat;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface ChatRepository extends MongoRepository<Chat, String> {

    Page<Chat> findAllByRoomId(Long roomId, Pageable pageable);

    List<Chat> findAllByRoomIdAndCreatedAtAfter(Long id, LocalDateTime lastCheckTime);

    Chat findFirstByRoomIdOrderByCreatedAtDesc(Long roomId);
}
