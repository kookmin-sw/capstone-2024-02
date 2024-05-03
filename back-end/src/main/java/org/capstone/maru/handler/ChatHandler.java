package org.capstone.maru.handler;

import io.jsonwebtoken.MalformedJwtException;
import java.util.Objects;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.exception.RestErrorCode;
import org.capstone.maru.security.exception.InvalidTokenException;
import org.capstone.maru.security.token.TokenProvider;
import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.Message;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.simp.stomp.StompCommand;
import org.springframework.messaging.simp.stomp.StompHeaderAccessor;
import org.springframework.messaging.support.ChannelInterceptor;
import org.springframework.messaging.support.MessageHeaderAccessor;

@Configuration
@RequiredArgsConstructor
@Slf4j
public class ChatHandler implements ChannelInterceptor {

    private final TokenProvider tokenProvider;

    @Override
    public Message<?> preSend(Message<?> message, MessageChannel channel) {

        log.info("preSend : {}", message);

        StompHeaderAccessor headerAccessor = MessageHeaderAccessor.getAccessor(message,
            StompHeaderAccessor.class);

        assert headerAccessor != null;

        String authorizationHeader = String.valueOf(
            headerAccessor.getNativeHeader("Authorization")
        );

        StompCommand command = headerAccessor.getCommand();

        log.info("authorizationHeader : {}", authorizationHeader);

        log.info("stompCommand : {}", headerAccessor.getCommand());
        
        if (!Objects.equals(command, StompCommand.CONNECT)) {
            return message;
        }

        if (authorizationHeader == null) {
            log.info("chat header가 없는 요청입니다.");
            throw new MalformedJwtException("jwt");
        }

        String token = "";
        String authorizationHeaderStr = authorizationHeader.replace("[", "").replace("]", "");

        if (authorizationHeaderStr.startsWith("Bearer ")) {
            token = authorizationHeaderStr.replace("Bearer ", "");

            // 권한 설정을 해야한다면 이곳에서
        } else {
            log.error("Authorization 헤더 형식이 틀리거나 없습니다. : {}", authorizationHeader);
            throw new MalformedJwtException("jwt");
        }

        if (!tokenProvider.validate(token)) {
            throw new InvalidTokenException(RestErrorCode.INVALID_TOKEN_VALUE);
        }

        return message;
    }
}
