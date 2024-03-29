package org.capstone.maru.security.token;

import io.jsonwebtoken.JwtException;
import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken;

public interface TokenProvider {

    boolean validate(String token);

    OAuth2AuthenticationToken decode(String token);

    TokenDto reissueAccessTokenUsing(String refreshToken) throws JwtException;

    TokenDto reissueTokensUsing(String refreshToken) throws JwtException;

    String createAccessToken(Authentication authentication);

    String createRefreshToken(Authentication authentication);

    Long getExpiration();
}
