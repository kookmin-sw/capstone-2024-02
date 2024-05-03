package org.capstone.maru.service;

import jakarta.transaction.Transactional;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.buffer.MessageBuffer;
import org.capstone.maru.domain.Chat;
import org.capstone.maru.domain.ChatRoom;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.MemberRoom;
import org.capstone.maru.dto.ChatMessage;
import org.capstone.maru.dto.response.ChatMemberProfileResponse;
import org.capstone.maru.dto.response.ChatMessageResponse;
import org.capstone.maru.dto.response.ChatRoomResponse;
import org.capstone.maru.repository.mongodb.ChatRepository;
import org.capstone.maru.repository.postgre.ChatRoomRepository;
import org.capstone.maru.repository.postgre.MemberAccountRepository;
import org.capstone.maru.repository.postgre.MemberRoomRepository;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.redis.core.ListOperations;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

@Slf4j
@RequiredArgsConstructor
@Service
public class ChatService {

    private static final String CHAT_ROOM_PREFIX = "chat:room:";

    private final ChatRepository chatRepository;

    private final MemberAccountRepository memberAccountRepository;

    private final ChatRoomRepository chatRoomRepository;

    private final MemberRoomRepository memberRoomRepository;

    private final MessageBuffer messageBuffer;

    private final RedisTemplate<String, String> redisTemplate;

    private final S3FileService s3FileService;

    @Transactional
    public void createChat(ChatMessage message) {
        // 채팅 메세지 버퍼와 Redis에 채팅 메세지 저장
        String messageId = UUID.randomUUID().toString();

        saveChatMessageRedis(messageId, message);

        messageBuffer.addMessage(Chat.from(messageId, message));
    }

    /*
    Redis에 채팅 메시지 저장
     */
    @Transactional
    public void saveChatMessageRedis(String messageId, ChatMessage message) {

        // Redis의 Hash 데이터 구조를 사용하여 채팅 메시지 저장
        ListOperations<String, String> listOperations = redisTemplate.opsForList();

        log.info("messageId : {}", messageId);

        String chatKey = CHAT_ROOM_PREFIX + message.roomId();

        listOperations.rightPushAll(chatKey, messageId, message.sender(), message.message(),
            message.createdAt().toString());

        redisTemplate.expire(chatKey, 1, java.util.concurrent.TimeUnit.DAYS); // 1일 뒤 만료
    }

    /*
    채팅방 입장시 채팅방의 메시지 조회
     */
    @Transactional
    public List<ChatMessageResponse> getChatMessages(Long roomId, int size, int page) {

        Comparator<ChatMessageResponse> comparator = Comparator.comparing(
            ChatMessageResponse::createdAt).reversed();

        // Redis의 Hash 데이터 구조에서 해당 채팅방의 모든 메시지 조회
        ListOperations<String, String> hashOperations = redisTemplate.opsForList();

        String chatKey = CHAT_ROOM_PREFIX + roomId.toString();

        log.info("chatKey : {}", chatKey);

        List<ChatMessageResponse> recentMessage = ChatMessageResponse.from(
            hashOperations.range(chatKey, 0, -1));

        List<String> messageId = recentMessage.stream().map(
            ChatMessageResponse::messageId
        ).toList();

        log.info("messageId : {}", messageId);

        /*
        MongoDB에서 채팅방의 메시지 조회, 최근 메시지가 먼저 조회되도록 정렬
         */
        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());

        chatRepository.findAllByRoomId(roomId, pageable).stream().map(
                chat -> {
                    if (messageId.contains(chat.getId())) {
                        return null;
                    } else {
                        return ChatMessageResponse.builder()
                            .messageId(chat.getId())
                            .sender(chat.getCreatedBy())
                            .message(chat.getMessage())
                            .createdAt(chat.getCreatedAt())
                            .build();
                    }
                }
            )
            .filter(Objects::nonNull)
            .forEach(recentMessage::add);

        recentMessage.sort(comparator);

        return recentMessage;
    }

    /*
    채팅방 생성
     */
    @Transactional
    public ChatRoom createChatRoom(String publisher, String name, List<String> members) {
        members.add(publisher);
        ChatRoom room = ChatRoom.createRoom(name);

        log.info("room : {}", room.getId());

        // 채팅방 생성 후 채팅방에 멤버 추가
        members.forEach(member -> {
            MemberAccount memberAccount = memberAccountRepository.findById(member).get();

            MemberRoom memberRoom = MemberRoom.createMemberRoom(memberAccount, room);
            memberRoomRepository.save(memberRoom);
        });

        log.info("finish create chat room");

        return chatRoomRepository.save(room);
    }

    /*
    채팅방의 멤버 보여주기
     */
    @Transactional
    public List<ChatMemberProfileResponse> showChatRoomMember(Long roomId) {
        ChatRoom room = chatRoomRepository.findById(roomId).orElseThrow(
            () -> new IllegalArgumentException("존재하지 않는 채팅방입니다.")
        );

        return room.getRoomMembers().stream().map(
            memberRoom -> {
                MemberAccount memberAccount = memberRoom.getMember();

                String profileImageUrl = s3FileService.getPreSignedUrlForLoad(
                    memberAccount.getProfileImage().getFileName());

                return ChatMemberProfileResponse.from(
                    memberAccount.getMemberId(),
                    memberAccount.getNickname(),
                    profileImageUrl
                );
            }
        ).toList();

    }

    /*
    채팅방에 멤버 추가하기
     */
    @Transactional
    public void addChatRoomMember(Long roomId, List<String> members) {
        ChatRoom room = chatRoomRepository.findById(roomId).orElseThrow(
            () -> new IllegalArgumentException("존재하지 않는 채팅방입니다.")
        );

        members.forEach(
            member -> {
                Optional<MemberAccount> memberAccount = memberAccountRepository.findByMemberId(
                    member);
                if (memberAccount.isEmpty()) {
                    throw new IllegalArgumentException("존재하지 않는 회원입니다.");
                }

                MemberRoom memberRoom = MemberRoom.createMemberRoom(memberAccount.get(), room);
                memberRoomRepository.save(memberRoom);
            }
        );

    }

    /*
    채팅방 리스트 보여주기
     */
    public List<ChatRoomResponse> showChatRoom(String memberId) {

        MemberAccount memberAccount = memberAccountRepository.findById(memberId).orElseThrow(
            () -> new IllegalArgumentException("존재하지 않는 회원입니다.")
        );

        log.info("memberAccount : {}", memberAccount.getChatRooms().get(0).getId());

        return memberAccount.getChatRooms().stream().map(
            chatRoom -> ChatRoomResponse.from(
                chatRoom.getChatRoom(),
                countUnreadChat(chatRoom),
                getLastMessage(chatRoom.getChatRoom())
            )
        ).toList();
    }

    /*
    redis와 mongo에서 현재 시간보다 이후의 채팅 개수 세기
     */
    public Integer countUnreadChat(MemberRoom memberRoom) {
        int unreadChatCount = 0;
        ChatRoom room = memberRoom.getChatRoom();

        // Redis에서 현재 시간보다 이후의 채팅 개수 세기
        List<String> chatMessages = redisTemplate.opsForList().range(
            CHAT_ROOM_PREFIX + room.getId(), 0, -1);

        if (chatMessages != null) {
            List<ChatMessageResponse> messages = ChatMessageResponse.from(chatMessages);

            unreadChatCount = (int) messages.stream().filter(
                message -> message.createdAt().isAfter(memberRoom.getLastCheckTime())
            ).count();

            log.info("unreadChatCount : {}", unreadChatCount);
        }

        // 현재 시간보다 이후의 채팅 개수 세기 in mongo
        List<Chat> chats = chatRepository.findAllByRoomIdAndCreatedAtAfter(room.getId(),
            memberRoom.getLastCheckTime());

        return unreadChatCount + chats.size();
    }

    /*
    채팅방의 마지막 메시지 조회
     */
    public ChatMessageResponse getLastMessage(ChatRoom room) {

        Long roomId = room.getId();
        // Redis에서 채팅방의 마지막 메시지 조회
        List<String> chatMessages = redisTemplate.opsForList().range(
            CHAT_ROOM_PREFIX + roomId, 0, -1);

        if (chatMessages != null) {
            List<ChatMessageResponse> messages = ChatMessageResponse.from(chatMessages);

            if (!messages.isEmpty()) {
                // 최신 메시지 반환
                return messages.get(messages.size() - 1);
            }
        }

        // MongoDB에서 채팅방의 마지막 메시지 조회
        Chat chat = chatRepository.findFirstByRoomIdOrderByCreatedAtDesc(roomId);

        return ChatMessageResponse.from(chat);
    }

    /*
     * 채팅방 나가는 경우
     */
    public void exitChatRoom(Long roomId, String memberId) {
        MemberRoom memberRoom = memberRoomRepository.findByMemberIdAndChatRoomId(memberId, roomId);

        memberRoom.updateLastCheckTime();
    }
}
