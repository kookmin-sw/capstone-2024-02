package org.capstone.maru.dto.response;

import lombok.Builder;
import org.capstone.maru.domain.ChatRoom;

@Builder
public record ChatRoomResponse(
    Long roomId,
    String roomName,
    int unreadCount,
    String lastMessage,
    String lastMessageTime
) {

    public static ChatRoomResponse from(Long roomId, String roomName, int unreadCount,
        String lastMessage, String lastMessageTime) {
        return ChatRoomResponse
            .builder()
            .roomId(roomId)
            .roomName(roomName)
            .unreadCount(unreadCount)
            .lastMessage(lastMessage)
            .lastMessageTime(lastMessageTime)
            .build();
    }

    public static ChatRoomResponse from(ChatRoom chatRoom, int unreadCount,
        ChatMessageResponse lastMessage) {

        return ChatRoomResponse
            .builder()
            .roomId(chatRoom.getId())
            .roomName(chatRoom.getName())
            .unreadCount(unreadCount)
            .lastMessage(lastMessage.message())
            .lastMessageTime(lastMessage.createdAt().toString())
            .build();
    }

}
