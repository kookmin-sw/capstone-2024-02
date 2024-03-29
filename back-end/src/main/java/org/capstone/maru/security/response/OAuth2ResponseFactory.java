package org.capstone.maru.security.response;

import java.util.Map;
import org.capstone.maru.security.constant.SocialType;
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;

public interface OAuth2ResponseFactory {

    /**
     * SocialType에 맞는 메소드 호출하여 OAuthAttributes 객체 반환 파라미터 : userNameAttributeName -> OAuth2 로그인 시
     * 키(PK)가 되는 값 / attributes : OAuth 서비스의 유저 정보들 소셜별 of 메소드(ofGoogle, ofKaKao, ofNaver)들은 각각 소셜
     * 로그인 API에서 제공하는 회원의 식별값(id), attributes, nameAttributeKey를 저장 후 build 필드가 들어갈 수 있을거 같은데
     */
    static OAuth2Response getOAuth2Response(String registrationId, Map<String, Object> attributes) {
        SocialType socialType = SocialType.of(registrationId);

        switch (socialType) {
            case KAKAO -> {
                return ofKakao(attributes);
            }
            case NAVER -> {
                return ofNaver(attributes);
            }
            default ->
                throw new OAuth2AuthenticationException("Unexpected socialType: " + socialType);
        }
    }

    static OAuth2Response getOAuth2Response(OAuth2AuthenticationToken oAuth2AuthenticationToken) {
        return getOAuth2Response(
            oAuth2AuthenticationToken.getAuthorizedClientRegistrationId(),
            oAuth2AuthenticationToken.getPrincipal().getAttributes()
        );
    }

    private static OAuth2Response ofKakao(Map<String, Object> attributes) {
        return KakaoOAuth2Response.from(attributes);
    }

    private static OAuth2Response ofNaver(Map<String, Object> attributes) {
        return NaverOAuth2Response.from(attributes);
    }
}
