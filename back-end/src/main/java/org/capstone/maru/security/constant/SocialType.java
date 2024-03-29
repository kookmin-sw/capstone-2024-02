package org.capstone.maru.security.constant;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
public enum SocialType {

    KAKAO("kakao"), NAVER("naver");

    private final String key;

    public static SocialType of(String registrationId) {
        return switch (registrationId) {
            case "kakao" -> SocialType.KAKAO;
            case "naver" -> SocialType.NAVER;
            default -> throw new IllegalArgumentException(
                "Unsupported social registrationId: " + registrationId);
        };
    }
}
