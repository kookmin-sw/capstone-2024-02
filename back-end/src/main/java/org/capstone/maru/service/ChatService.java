package org.capstone.maru.service;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.Chat;
import org.capstone.maru.domain.ChatRoom;
import org.capstone.maru.repository.ChatRepository;
import org.springframework.stereotype.Service;

@Slf4j
@RequiredArgsConstructor
@Service
public class ChatService {

    private final ChatRepository chatRepository;

    @Transactional
    public ChatRoom createChatRoom(String name) {
        ChatRoom room = ChatRoom.createRoom(name);
        log.info("ChatRoom created : {}, {}", room.getId(), room.getName());
        return chatRepository.save(room);
    }

    @Transactional
    public Chat createChat(Long roomId, String sender, String message) {
        ChatRoom room = chatRepository.findById(roomId).get();
        return Chat.createChat(room, sender, message);
    }
}
