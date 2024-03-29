package org.capstone.maru.security.token.jwt;

import jakarta.servlet.http.HttpServletRequest;
import org.capstone.maru.exception.RestErrorCode;
import org.capstone.maru.security.exception.InvalidTokenException;
import org.capstone.maru.security.exception.RefreshTokenNotFoundException;
import org.capstone.maru.security.token.RefreshTokenRepository;
import org.capstone.maru.security.token.RefreshTokenService;
import org.capstone.maru.security.token.TokenDto;
import org.capstone.maru.security.token.TokenProvider;
import org.capstone.maru.security.token.TokenReIssuer;
import org.capstone.maru.security.token.TokenResolver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

@Component("jwtTokenReIssuer")
public class JwtTokenReIssuer implements TokenReIssuer {

    private final TokenProvider tokenProvider;
    private final TokenResolver tokenResolver;

    private final RefreshTokenService refreshTokenService;

    public JwtTokenReIssuer(
        @Qualifier("jwtTokenProvider") TokenProvider tokenProvider,
        @Qualifier("jwtTokenResolver") TokenResolver tokenResolver,
        @Autowired RefreshTokenService refreshTokenService
    ) {
        this.tokenProvider = tokenProvider;
        this.tokenResolver = tokenResolver;
        this.refreshTokenService = refreshTokenService;
    }

    @Override
    public TokenDto reissueAccessToken(HttpServletRequest request) throws InvalidTokenException {
        String refreshToken = tokenResolver.resolveTokenOrNull(request);

        if (!tokenProvider.validate(refreshToken)) {
            throw new InvalidTokenException(RestErrorCode.INVALID_TOKEN_VALUE, "유효하지 않은 토큰입니다.");
        }

        if (!refreshTokenService.hasRefreshToken(refreshToken)) {
            throw new RefreshTokenNotFoundException(RestErrorCode.NOT_FOUND);
        }

        return tokenProvider.reissueAccessTokenUsing(refreshToken);
    }

    @Override
    public TokenDto reissueTokens(HttpServletRequest request) throws InvalidTokenException {
        String refreshToken = tokenResolver.resolveTokenOrNull(request);

        if (!tokenProvider.validate(refreshToken)) {
            throw new InvalidTokenException(RestErrorCode.INVALID_TOKEN_VALUE, "유효하지 않은 토큰입니다.");
        }

        if (!refreshTokenService.hasRefreshToken(refreshToken)) {
            throw new RefreshTokenNotFoundException(RestErrorCode.NOT_FOUND);
        }

        return tokenProvider.reissueTokensUsing(refreshToken);
    }
}
