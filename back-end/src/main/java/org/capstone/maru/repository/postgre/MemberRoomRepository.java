package org.capstone.maru.repository.postgre;

import java.util.Optional;
import org.capstone.maru.domain.MemberRoom;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MemberRoomRepository extends JpaRepository<MemberRoom, Long> {

    Optional<MemberRoom> findByMemberIdAndChatRoomId(String memberId, Long chatRoomId);

    void deleteByMemberIdAndChatRoomId(String memberId, Long chatRoomId);

    boolean existsByChatRoomId(Long chatRoomId);
}
