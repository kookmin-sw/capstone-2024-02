package org.capstone.maru.controller;

import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Pageable;
import org.capstone.maru.dto.request.ChatRoomMemberRequest;
import org.capstone.maru.dto.request.ChatRoomRequest;
import org.capstone.maru.dto.response.APIResponse;
import org.capstone.maru.dto.response.ChatMemberProfileResponse;
import org.capstone.maru.dto.response.ChatMessageResponse;
import org.capstone.maru.dto.response.ChatRoomResponse;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.service.ChatService;
import org.springframework.data.domain.Sort.Direction;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RequiredArgsConstructor
@RestController
@RequestMapping("/chatRoom")
public class ChatController {

    private final ChatService chatService;

    /*
    채팅방 생성
     */
    @PostMapping
    public ResponseEntity<APIResponse> chatRoom(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @RequestBody ChatRoomRequest roomRequest
    ) {
        log.info("roomRequest : {}, memberPrincipal : {}", roomRequest, memberPrincipal.memberId());
        // 채팅방 아이디 반환
        Long data = chatService.createChatRoom(memberPrincipal.memberId(), roomRequest.roomName(),
            roomRequest.members()).getId();

        return ResponseEntity.ok(APIResponse.success(data));
    }

    /*
    채팅방 멤버 보여주기
     */
    @GetMapping("/{roomId}")
    public ResponseEntity<APIResponse> showChatRoomMember(@PathVariable Long roomId) {
        List<ChatMemberProfileResponse> data = chatService.showChatRoomMember(roomId);
        return ResponseEntity.ok(APIResponse.success(data));
    }

    /*
    채팅방 보여주기
     */
    @GetMapping
    public ResponseEntity<APIResponse> showChatRoom(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal) {

        log.info("memberPrincipal : {}", memberPrincipal.memberId());

        List<ChatRoomResponse> data = chatService.showChatRoom(memberPrincipal.memberId());

        log.info("data : {}", data);
        return ResponseEntity.ok(APIResponse.success(data));
    }

    /*
    채팅방에 멤버 추가하기
     */
    @PostMapping("/{roomId}/invite")
    public ResponseEntity<APIResponse> addChatRoomMember(@PathVariable Long roomId,
        @RequestBody ChatRoomMemberRequest memberId) {
        log.info("roomId : {}, memberId : {}", roomId, memberId);

        chatService.addChatRoomMember(roomId, memberId.members());

        return ResponseEntity.ok(APIResponse.success("success"));
    }

    /*
    최근 채팅 조회
     */
    @GetMapping("/{roomId}/chat")
    public ResponseEntity<APIResponse> showChat(
        @PathVariable Long roomId,
        @PageableDefault(sort = "createdAt", direction = Direction.DESC) Pageable pageable
    ) {
        log.info("roomId : {}", roomId);
        log.info("pageable : {}", pageable);

        List<ChatMessageResponse> data = chatService.getChatMessages(roomId,
            pageable);

        log.info("{}", data);

        return ResponseEntity.ok(APIResponse.success(data));
    }

    /*
    채팅방 나가기
     */
    @PostMapping("/{roomId}/exit")
    public ResponseEntity<APIResponse> exitChatRoom(@PathVariable Long roomId,
        @AuthenticationPrincipal MemberPrincipal memberPrincipal) {
        log.info("roomId : {}, memberId : {}", roomId, memberPrincipal.memberId());

        chatService.exitChatRoom(roomId, memberPrincipal.memberId());

        return ResponseEntity.ok(APIResponse.success("success"));
    }
}

