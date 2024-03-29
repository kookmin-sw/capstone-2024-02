package org.capstone.maru.security.token;

import jakarta.servlet.http.HttpServletRequest;
import org.capstone.maru.security.exception.InvalidTokenException;

public interface TokenReIssuer {

    TokenDto reissueAccessToken(HttpServletRequest request) throws InvalidTokenException;

    TokenDto reissueTokens(HttpServletRequest request) throws InvalidTokenException;
}
