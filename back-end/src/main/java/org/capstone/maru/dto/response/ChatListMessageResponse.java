package org.capstone.maru.dto.response;

import java.time.LocalDateTime;
import lombok.Builder;

@Builder
public record ChatListMessageResponse(
    String messageId,
    Long roomId,
    String sender,
    String message,
    String nickname,
    LocalDateTime createdAt
) {

    public static ChatListMessageResponse from(ChatMessageResponse chatMessageResponse,
        Long roomId) {
        return ChatListMessageResponse.builder()
            .messageId(chatMessageResponse.messageId())
            .roomId(roomId)
            .sender(chatMessageResponse.sender())
            .message(chatMessageResponse.message())
            .nickname(chatMessageResponse.nickname())
            .createdAt(chatMessageResponse.createdAt())
            .build();
    }
}
