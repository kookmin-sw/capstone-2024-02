package org.capstone.maru.security.handler;

import static org.capstone.maru.security.cookie.HttpCookieOAuth2AuthorizationRequestRepository.REDIRECT_URL_PARAM_COOKIE_NAME;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.Cookie;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.security.cookie.CookieUtils;
import org.capstone.maru.security.cookie.HttpCookieOAuth2AuthorizationRequestRepository;
import org.capstone.maru.security.token.TokenProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.SimpleUrlAuthenticationSuccessHandler;
import org.springframework.stereotype.Component;
import org.springframework.web.util.UriComponentsBuilder;

@Slf4j
@Component("customOAuth2AuthenticationSuccessHandler")
public class CustomOAuth2AuthenticationSuccessHandler extends
    SimpleUrlAuthenticationSuccessHandler {

    private final TokenProvider tokenProvider;
    private final HttpCookieOAuth2AuthorizationRequestRepository httpCookieOAuth2AuthorizationRequestRepository;

    public CustomOAuth2AuthenticationSuccessHandler(
        @Qualifier("jwtTokenProvider") TokenProvider tokenProvider,
        @Autowired HttpCookieOAuth2AuthorizationRequestRepository httpCookieOAuth2AuthorizationRequestRepository
    ) {
        this.tokenProvider = tokenProvider;
        this.httpCookieOAuth2AuthorizationRequestRepository = httpCookieOAuth2AuthorizationRequestRepository;
    }

    @Override
    public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response,
        Authentication authentication) throws IOException {
        log.info("[Debug] oAuth2 Authentication Success!");

        String targetUrl = this.determineTargetUrl(request, response, authentication);

        if (response.isCommitted()) {
            logger.info(
                "[Debug] Response has already been committed. Unable to redirect to " + targetUrl);
            return;
        }

        this.clearAuthenticationAttributes(request);
        this.httpCookieOAuth2AuthorizationRequestRepository.clearCookies(request, response);
        this.getRedirectStrategy().sendRedirect(request, response, targetUrl);
    }

    @Override
    protected String determineTargetUrl(HttpServletRequest request, HttpServletResponse response,
        Authentication authentication) {
        // TODO: cookie에 access_token, refresh_token 넣어 보내기

        log.info("[Debug] DetermineTargetUrl!");

        String targetUrl = CookieUtils
            .resolveCookie(request, REDIRECT_URL_PARAM_COOKIE_NAME)
            .map(Cookie::getValue)
            .orElse("/");

        return UriComponentsBuilder
            .fromUriString(targetUrl)
            .queryParam("access_token", tokenProvider.createAccessToken(authentication))
            .queryParam("refresh_token", tokenProvider.createRefreshToken(authentication))
            .queryParam("expires_in", tokenProvider.getExpiration())
            .encode(StandardCharsets.UTF_8)
            .build().toUriString();
    }
}
