package org.capstone.maru.repository.postgre;

import org.capstone.maru.domain.MemberRoom;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MemberRoomRepository extends JpaRepository<MemberRoom, Long> {

    MemberRoom findByMemberIdAndChatRoomId(String memberId, Long chatRoomId);
}
