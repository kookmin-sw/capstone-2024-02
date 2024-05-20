package org.capstone.maru.controller;

import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.dto.ChatMessage;
import org.capstone.maru.dto.request.ChatMessageRequest;
import org.capstone.maru.dto.response.ChatListMessageResponse;
import org.capstone.maru.dto.response.ChatMemberProfileResponse;
import org.capstone.maru.dto.response.ChatMessageResponse;
import org.capstone.maru.service.ChatService;
import org.springframework.messaging.handler.annotation.DestinationVariable;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RequiredArgsConstructor
@RestController
public class SocketController {

    private final ChatService chatService;

    private final SimpMessagingTemplate simpMessagingTemplate;

    @MessageMapping("/{roomId}") //여기로 전송되면 메서드 호출 -> WebSocketConfig prefixes 에서 적용한건 앞에 생략
    @SendTo({"/room/{roomId}"})
    //구독하고 있는 장소로 메시지 전송 (목적지)  -> WebSocketConfig Broker 에서 적용한건 앞에 붙어줘야됨
    public ChatMessageResponse chat(@DestinationVariable Long roomId,
        @Payload ChatMessageRequest message) {

        log.info("roomId : {}, message room id : {}", roomId, message.roomId());

        if (!roomId.equals(message.roomId())) {
            throw new IllegalArgumentException("roomId is not matched");
        }

        log.info("roomId : {}, sender : {}, message : {}", roomId, message.sender(),
            message.message());

        //채팅 저장
        ChatMessageResponse chatMessageResponse = chatService.createChat(message);

        //채팅방에 있는 사람들 불러오기
        List<ChatMemberProfileResponse> chatRoomMemberProfileList =
            chatService.showChatRoomMember(roomId);

        List<String> roomMemberIdList = chatRoomMemberProfileList.stream()
            .map(ChatMemberProfileResponse::memberId).toList();

        // 채팅방에 있는 사람들에게 메시지 전송 채팅이 도착했다.
        roomMemberIdList.forEach(
            memberId -> simpMessagingTemplate.convertAndSend("/roomList/" + memberId,
                ChatListMessageResponse.from(chatMessageResponse, roomId))
        );

        return chatMessageResponse;
    }

}
