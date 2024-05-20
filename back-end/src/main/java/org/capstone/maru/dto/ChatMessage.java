package org.capstone.maru.dto;

import java.time.LocalDateTime;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.dto.request.ChatMessageRequest;

@Slf4j
@Builder
public record ChatMessage(
    String messageId,
    Long roomId,
    String sender,
    String message,
    String nickname,
    LocalDateTime createdAt
) {

    public static ChatMessage createChatMessage(Long roomId, String sender, String message) {
        return ChatMessage.builder()
            .roomId(roomId)
            .sender(sender)
            .message(message)
            .build();
    }

    public static ChatMessage currentMessageFrom(String messageId, ChatMessageRequest message) {
        return ChatMessage.builder()
            .messageId(messageId)
            .roomId(message.roomId())
            .sender(message.sender())
            .message(message.message())
            .nickname(message.nickname())
            .createdAt(LocalDateTime.now())
            .build();
    }

}
