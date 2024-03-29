package org.capstone.maru.security.token.jwt;

import jakarta.servlet.http.HttpServletRequest;
import org.capstone.maru.security.token.TokenResolver;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

@Component("jwtTokenResolver")
public class JwtTokenResolver implements TokenResolver {

    private static final String AUTHORIZATION_HEADER = "Authorization";

    @Override
    public String resolveTokenOrNull(HttpServletRequest request) {
        String bearerToken = request.getHeader(AUTHORIZATION_HEADER);

        if (StringUtils.hasText(bearerToken) && bearerToken.startsWith("Bearer ")) {
            return bearerToken.substring(7);
        }

        return null;
    }
}
