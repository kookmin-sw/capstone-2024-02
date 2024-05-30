package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.OneToMany;
import java.util.List;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@Entity
public class ChatRoom {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "chatRoom_id")
    private Long id;

    private String name;

    // 채팅방 참여자
    @OneToMany(mappedBy = "chatRoom", fetch = FetchType.EAGER, orphanRemoval = true)
    private List<MemberRoom> roomMembers;

    @Builder
    public ChatRoom(String name) {
        this.name = name;
    }

    public static ChatRoom createRoom(String name) {
        return ChatRoom.builder()
                       .name(name)
                       .build();
    }

    public void updateChatRoomName(String name) {
        this.name = name;
    }
}
