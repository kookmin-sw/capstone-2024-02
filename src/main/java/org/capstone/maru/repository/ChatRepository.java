package org.capstone.maru.repository;

import org.capstone.maru.domain.Chat;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface ChatRepository extends MongoRepository<Chat, Long> {

}
