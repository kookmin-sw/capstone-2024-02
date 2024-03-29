package org.capstone.maru.security.response;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import lombok.Builder;
import lombok.Getter;
import org.capstone.maru.domain.constant.Gender;

/**
 * https://developers.kakao.com/docs/latest/ko/kakaologin/rest-api#req-user-info 카카오로부터 사용자 정보를 불러올
 * 때 카카오에서 응답 메시지 형식을 맞춘 것이다. 자세한 내용은 위 링크를 참조
 */
@SuppressWarnings("unchecked")
@Getter
public class KakaoOAuth2Response extends OAuth2Response {

    private final Long id;
    private final LocalDateTime connectedAt;
    private final Map<String, Object> properties;
    private final KakaoAccount kakaoAccount;

    @Builder
    public record KakaoAccount(
        Boolean profileNicknameNeedsAgreement,
        Profile profile,
        Boolean hasEmail,
        Boolean emailNeedsAgreement,
        Boolean isEmailValid,
        Boolean isEmailVerified,
        String email,
        Boolean birthyearNeedsAgreement,
        String birthyear,
        Boolean genderNeedsAgreement,
        String gender,
        Boolean phoneNumberNeedsAgreement,
        String phoneNumber
    ) {

        public record Profile(String nickname) {

            public static Profile from(Map<String, Object> attributes) {
                return new Profile(String.valueOf(attributes.get("nickname")));
            }
        }

        public static KakaoAccount from(Map<String, Object> attributes) {
            return KakaoAccount.builder()
                .profileNicknameNeedsAgreement(
                    Boolean.valueOf(String.valueOf(
                        attributes.get("profile_nickname_needs_agreement"))
                    )
                )
                .profile(
                    Profile.from((Map<String, Object>) attributes.get("profile"))
                )
                .hasEmail(
                    Boolean.valueOf(String.valueOf(attributes.get("has_email")))
                )
                .emailNeedsAgreement(
                    Boolean.valueOf(
                        String.valueOf(attributes.get("email_needs_agreement"))
                    )
                )
                .isEmailValid(
                    Boolean.valueOf(
                        String.valueOf(attributes.get("is_email_valid"))
                    )
                )
                .isEmailVerified(
                    Boolean.valueOf(
                        String.valueOf(attributes.get("is_email_verified"))
                    )
                )
                .email(String.valueOf(attributes.get("email")))
                .birthyearNeedsAgreement(
                    Boolean.valueOf(
                        String.valueOf(attributes.get("birthyear_needs_agreement"))
                    )
                )
                .birthyear(String.valueOf(attributes.get("birthyear")))
                .genderNeedsAgreement(
                    Boolean.valueOf(
                        String.valueOf(attributes.get("gender_needs_agreement"))
                    )
                )
                .gender(String.valueOf(attributes.get("gender")))
                .phoneNumberNeedsAgreement(
                    Boolean.valueOf(
                        String.valueOf(
                            attributes.get("phone_number_needs_agreement"))
                    )
                )
                .phoneNumber(String.valueOf(attributes.get("phone_number")))
                .build();
        }

        public String nickname() {
            return this.profile().nickname();
        }
    }

    private KakaoOAuth2Response(Long id, LocalDateTime connectedAt, Map<String, Object> properties,
        KakaoAccount kakaoAccount) {
        this.id = id;
        this.connectedAt = connectedAt;
        this.properties = properties;
        this.kakaoAccount = kakaoAccount;
    }

    public static KakaoOAuth2Response from(Map<String, Object> attributes) {
        return new KakaoOAuth2Response(
            Long.valueOf(String.valueOf(attributes.get("id"))),
            LocalDateTime.parse(
                String.valueOf(attributes.get("connected_at")),
                DateTimeFormatter.ISO_INSTANT.withZone(ZoneId.systemDefault())
            ),
            (Map<String, Object>) attributes.get("properties"),
            KakaoAccount.from((Map<String, Object>) attributes.get("kakao_account"))
        );
    }

    // -- OAuth2Response abstract method 구현 -- //
    @Override
    public String id() {
        return String.valueOf(this.id);
    }

    @Override
    public String email() {
        return this.getKakaoAccount().email();
    }

    @Override
    public String nickname() {
        return this.getKakaoAccount().nickname();
    }

    @Override
    public String birthYear() {
        return this.getKakaoAccount().birthyear;
    }

    @Override
    public Gender gender() {
        if (this.getKakaoAccount().gender().equals("female")) {
            return Gender.FEMALE;
        }
        return Gender.MALE;
    }

    @Override
    public String phoneNumber() {
        return this.getKakaoAccount().phoneNumber();
    }
}
