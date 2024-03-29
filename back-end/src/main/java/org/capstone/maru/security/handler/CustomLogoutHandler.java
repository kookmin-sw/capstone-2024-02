package org.capstone.maru.security.handler;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.capstone.maru.security.token.TokenRemover;
import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.logout.LogoutHandler;
import org.springframework.stereotype.Component;

@RequiredArgsConstructor
@Component("customLogoutHandler")
public class CustomLogoutHandler implements LogoutHandler {

    private final TokenRemover tokenRemover;

    @Override
    public void logout(HttpServletRequest request, HttpServletResponse response,
        Authentication authentication) {
        tokenRemover.removeRefreshToken(request);

        // TODO: 로그아웃 시 해야할 것들 작성
    }
}
