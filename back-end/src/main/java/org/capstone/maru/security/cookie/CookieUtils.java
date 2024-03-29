package org.capstone.maru.security.cookie;

import com.google.gson.Gson;
import io.micrometer.common.util.StringUtils;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.util.Arrays;
import java.util.Base64;
import java.util.Objects;
import java.util.Optional;
import org.springframework.security.oauth2.core.endpoint.OAuth2AuthorizationRequest;

public interface CookieUtils {

    static Optional<Cookie> resolveCookie(
        HttpServletRequest request,
        String cookieName
    ) {
        Cookie[] cookies = request.getCookies();

        if (cookies == null) {
            return Optional.empty();
        }

        return Arrays.stream(cookies)
                     .filter(cookie -> cookie.getName().equals(cookieName))
                     .map(Optional::of)
                     .findFirst()
                     .orElse(Optional.empty());
    }

    static void deleteCookie(
        HttpServletRequest request,
        HttpServletResponse response,
        String cookieName
    ) {
        Cookie[] cookies = request.getCookies();

        if (cookies == null) {
            return;
        }

        for (Cookie cookie : cookies) {
            if (cookie.getName().equals(cookieName)) {
                cookie.setValue("");
                cookie.setPath("/");
                cookie.setMaxAge(0);
                response.addCookie(cookie);
                break;
            }
        }
    }

    static void setCookie(HttpServletResponse response, String cookieName, String cookieContents,
        int maxAge) {
        Cookie cookie = new Cookie(cookieName, cookieContents);
        cookie.setPath("/");
        cookie.setHttpOnly(true);
        cookie.setMaxAge(maxAge);
        response.addCookie(cookie);
    }

    static String serialize(OAuth2AuthorizationRequest request) {
        Gson gson = new Gson();
        return Base64.getUrlEncoder()
                     .encodeToString(gson.toJson(request).getBytes());
    }

    static <T> T deserialize(Cookie cookie, Class<T> clz) {
        if (isDeleted(cookie)) {
            return null;
        } else {
            Gson gson = new Gson();
            return gson
                .fromJson(
                    new String(
                        Base64.getUrlDecoder()
                              .decode(cookie.getValue())
                    ),
                    clz
                );
        }
    }

    private static boolean isDeleted(Cookie cookie) {
        return StringUtils.isBlank(cookie.getValue()) || Objects.isNull(cookie.getValue());
    }
}
