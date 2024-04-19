package org.capstone.maru.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.Chat;
import org.capstone.maru.domain.ChatRoom;
import org.capstone.maru.dto.request.ChatMessageRequest;
import org.capstone.maru.service.ChatService;
import org.springframework.messaging.handler.annotation.DestinationVariable;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RequiredArgsConstructor
@RestController
public class ChatController {

    private final ChatService chatService;

    @GetMapping("/chatRoom/{roomName}")
    public ChatRoom chatRoom(@PathVariable String roomName) {
        return chatService.createChatRoom(roomName);
    }

    @MessageMapping("/{roomId}") //여기로 전송되면 메서드 호출 -> WebSocketConfig prefixes 에서 적용한건 앞에 생략
    @SendTo("/room/{roomId}")
    //구독하고 있는 장소로 메시지 전송 (목적지)  -> WebSocketConfig Broker 에서 적용한건 앞에 붙어줘야됨
    public ChatMessageRequest chat(@DestinationVariable Long roomId,
        @Payload ChatMessageRequest message) {

        log.info("roomId : {}, message room id : {}", roomId, message.roomId());

        if (!roomId.equals(message.roomId())) {
            throw new IllegalArgumentException("roomId is not matched");
        }

        log.info("roomId : {}, sender : {}, message : {}", roomId, message.sender(),
            message.message());

        //채팅 저장
        Chat chat = chatService.createChat(roomId, message.sender(), message.message());
        return ChatMessageRequest.builder()
            .roomId(roomId)
            .sender(chat.getSender())
            .message(chat.getMessage())
            .build();
    }
}

