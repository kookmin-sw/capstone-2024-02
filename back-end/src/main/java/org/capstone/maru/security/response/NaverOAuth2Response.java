package org.capstone.maru.security.response;

import java.util.Map;
import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.constant.Gender;

@SuppressWarnings("unchecked")
@Getter
@Slf4j
public class NaverOAuth2Response extends OAuth2Response {

    private final Response response;

    @Builder
    private record Response(
        String id,
        String email,
        String name,
        String birthyear,
        String gender,
        String phoneNumber
    ) {

        public static Response from(Map<String, Object> attributes) {
            return Response.builder()
                .id(String.valueOf(attributes.get("id")))
                .email(String.valueOf(attributes.get("email")))
                .name(String.valueOf(attributes.get("name")))
                .birthyear(String.valueOf(attributes.get("birthyear")))
                .gender(String.valueOf(attributes.get("gender")))
                .phoneNumber(String.valueOf(attributes.get("mobile")))
                .build();
        }
    }

    private NaverOAuth2Response(Response response) {
        this.response = response;
    }

    public static NaverOAuth2Response from(Map<String, Object> attributes) {
        log.info("attributes: {}", attributes);
        return new NaverOAuth2Response(
            Response.from((Map<String, Object>) attributes.get("response")));
    }

    @Override
    public String id() {
        return response.id();
    }

    @Override
    public String email() {
        return response.email();
    }

    @Override
    public String nickname() {
        return response.name();
    }

    @Override
    public String birthYear() {
        return this.response.birthyear;
    }

    @Override
    public Gender gender() {
        if (this.response.gender.equals("F")) {
            return Gender.FEMALE;
        }
        return Gender.MALE;
    }

    @Override
    public String phoneNumber() {
        return this.response.phoneNumber;
    }
}