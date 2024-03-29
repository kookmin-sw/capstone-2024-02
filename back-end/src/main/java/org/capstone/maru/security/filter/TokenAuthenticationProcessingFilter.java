package org.capstone.maru.security.filter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.exception.RestErrorCode;
import org.capstone.maru.security.exception.InvalidTokenException;
import org.capstone.maru.security.handler.TokenAuthenticationFailureHandler;
import org.capstone.maru.security.token.TokenProvider;
import org.capstone.maru.security.token.TokenResolver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.OncePerRequestFilter;

@Slf4j
@Component("tokenAuthenticationProcessingFilter")
public class TokenAuthenticationProcessingFilter extends OncePerRequestFilter {

    private final TokenAuthenticationFailureHandler failureHandler;
    private final TokenProvider tokenProvider;
    private final TokenResolver tokenResolver;

    public TokenAuthenticationProcessingFilter(
        @Autowired TokenAuthenticationFailureHandler failureHandler,
        @Qualifier("jwtTokenProvider") TokenProvider tokenProvider,
        @Qualifier("jwtTokenResolver") TokenResolver tokenResolver
    ) {
        this.failureHandler = failureHandler;
        this.tokenProvider = tokenProvider;
        this.tokenResolver = tokenResolver;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
        FilterChain filterChain) throws ServletException, IOException {

        log.info("[Debug] Token Auth Filter running");

        String token = tokenResolver.resolveTokenOrNull(request);

        if (SecurityContextHolder.getContext().getAuthentication() == null && StringUtils.hasText(
            token)) {
            try {
                if (!tokenProvider.validate(token)) {
                    throw new InvalidTokenException(RestErrorCode.INVALID_TOKEN_VALUE);
                }

                Authentication authentication = tokenProvider.decode(token);

                log.info("[Debug] createToken or useToken!");

                SecurityContextHolder.getContext().setAuthentication(authentication);

            } catch (InvalidTokenException ex) {
                this.unsuccessfulAuthentication(request, response, ex);
                return;
            }
        }

        filterChain.doFilter(request, response);
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) throws ServletException {
        return request.getRequestURI().contains("token/");
    }

    protected void unsuccessfulAuthentication(HttpServletRequest request,
        HttpServletResponse response,
        AuthenticationException failed) throws IOException, ServletException {

        SecurityContextHolder.clearContext();
        log.info("[Debug] jwt token validation fail", failed);
        log.info("[Debug] Cleared SecurityContextHolder");
        log.info("[Debug] Handling authentication failure");
        this.failureHandler.onAuthenticationFailure(request, response, failed);
    }
}
