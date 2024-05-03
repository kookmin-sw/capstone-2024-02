package org.capstone.maru.domain;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import java.time.LocalDateTime;
import lombok.Getter;
import lombok.NoArgsConstructor;

/*
   member 와 chatRoom의 연관관계 table
 */
@Getter
@NoArgsConstructor
@Entity
public class MemberRoom {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "member_id")
    private MemberAccount member;

    @ManyToOne
    @JoinColumn(name = "chatRoom_id")
    private ChatRoom chatRoom;

    private LocalDateTime lastCheckTime;

    /*
     추가적인 column이 필요할 경우 추가
     */

    public MemberRoom(MemberAccount member, ChatRoom chatRoom, LocalDateTime lastCheckTime) {
        this.member = member;
        this.chatRoom = chatRoom;
        this.lastCheckTime = lastCheckTime;
    }

    public static MemberRoom createMemberRoom(MemberAccount memberAccount, ChatRoom room) {
        return new MemberRoom(memberAccount, room, LocalDateTime.now());
    }

    /*
       disconnect 시에 마지막 확인 시간을 업데이트
     */
    public void updateLastCheckTime() {
        this.lastCheckTime = LocalDateTime.now();
    }
}
