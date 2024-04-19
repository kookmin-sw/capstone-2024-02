package org.capstone.maru.security.handler;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.security.token.TokenRemover;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.logout.LogoutHandler;
import org.springframework.security.web.authentication.logout.LogoutSuccessHandler;
import org.springframework.security.web.authentication.logout.SimpleUrlLogoutSuccessHandler;
import org.springframework.stereotype.Component;

@Slf4j
@RequiredArgsConstructor
@Component("customLogoutHandler")
public class CustomLogoutSuccessHandler implements LogoutSuccessHandler {

    private final TokenRemover tokenRemover;

    @Override
    public void onLogoutSuccess(HttpServletRequest request, HttpServletResponse response,
        Authentication authentication)
        throws IOException, ServletException {
        log.info("[Debug] logout success handler");

        // token 삭제
        tokenRemover.removeRefreshToken(request);

        // 로그 아웃 성공 시 해야할 일
        response.setStatus(HttpStatus.OK.value());
    }
}
