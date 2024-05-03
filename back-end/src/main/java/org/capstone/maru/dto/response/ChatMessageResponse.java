package org.capstone.maru.dto.response;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.Chat;
import org.capstone.maru.dto.ChatMessage;

@Slf4j
@Builder
public record ChatMessageResponse(
    String messageId,
    String sender,
    String message,
    LocalDateTime createdAt

) {

    public static List<ChatMessageResponse> from(List<String> messages) {
        List<ChatMessageResponse> chatMessages = new ArrayList<>();

        log.info("messages : {}", messages);

        for (int i = 0; i < messages.size(); i += 4) {
            chatMessages.add(ChatMessageResponse.builder()
                .messageId(messages.get(i))
                .sender(messages.get(i + 1))
                .message(messages.get(i + 2))
                .createdAt(LocalDateTime.parse(messages.get(i + 3)))
                .build());
        }

        log.info("chatMessages : {}", chatMessages);
        return chatMessages;
    }

    public static ChatMessageResponse from(Chat chat) {
        return ChatMessageResponse.builder()
            .messageId(chat.getId())
            .sender(chat.getCreatedBy())
            .message(chat.getMessage())
            .createdAt(chat.getCreatedAt())
            .build();
    }

}
