package org.capstone.maru.security.token.jwt;

import jakarta.servlet.http.HttpServletRequest;
import org.capstone.maru.security.token.RefreshTokenService;
import org.capstone.maru.security.token.TokenProvider;
import org.capstone.maru.security.token.TokenRemover;
import org.capstone.maru.security.token.TokenResolver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

@Component("jwtTokenRemover")
public class JwtTokenRemover implements TokenRemover {

    private final RefreshTokenService refreshTokenService;
    private final TokenProvider tokenProvider;
    private final TokenResolver tokenResolver;

    public JwtTokenRemover(
        @Autowired RefreshTokenService refreshTokenService,
        @Qualifier("jwtTokenProvider") TokenProvider tokenProvider,
        @Qualifier("jwtTokenResolver") TokenResolver tokenResolver
    ) {
        this.refreshTokenService = refreshTokenService;
        this.tokenProvider = tokenProvider;
        this.tokenResolver = tokenResolver;
    }

    @Override
    public void removeRefreshToken(HttpServletRequest request) {
        String refreshToken = tokenResolver.resolveTokenOrNull(request);

        if (!tokenProvider.validate(refreshToken)) {
            return;
        }

        refreshTokenService.removeRefreshToken(refreshToken);
    }
}
