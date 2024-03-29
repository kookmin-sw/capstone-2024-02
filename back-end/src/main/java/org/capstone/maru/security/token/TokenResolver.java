package org.capstone.maru.security.token;

import jakarta.servlet.http.HttpServletRequest;

public interface TokenResolver {

    String resolveTokenOrNull(HttpServletRequest request);
}
