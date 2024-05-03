package org.capstone.maru.listener;

import java.util.Objects;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.repository.postgre.MemberAccountRepository;
import org.capstone.maru.repository.postgre.MemberRoomRepository;
import org.springframework.context.ApplicationListener;
import org.springframework.messaging.simp.stomp.StompHeaderAccessor;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.messaging.SessionDisconnectEvent;

@RequiredArgsConstructor
@Component
@Slf4j
public class DisconnectEventListener implements ApplicationListener<SessionDisconnectEvent> {

    private final MemberRoomRepository memberRoomRepository;

    private final MemberAccountRepository memberAccountRepository;

    @Override
    public void onApplicationEvent(SessionDisconnectEvent event) {
        StompHeaderAccessor headerAccessor = StompHeaderAccessor.wrap(event.getMessage());

        String memberId = Objects.requireNonNull(headerAccessor.getSessionAttributes())
            .get("memberId").toString();

        Integer roomId = (Integer) Objects.requireNonNull(headerAccessor.getSessionAttributes())
            .get("roomId");

        log.info("memberId : {}", memberId);
        log.info("roomId : {}", roomId);
    }
}
