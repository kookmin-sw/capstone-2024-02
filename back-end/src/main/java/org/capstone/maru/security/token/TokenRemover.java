package org.capstone.maru.security.token;

import jakarta.servlet.http.HttpServletRequest;

public interface TokenRemover {

    void removeRefreshToken(HttpServletRequest request);
}
