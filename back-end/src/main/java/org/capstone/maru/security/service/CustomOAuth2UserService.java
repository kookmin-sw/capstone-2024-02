package org.capstone.maru.security.service;

import java.util.Map;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.security.response.OAuth2Response;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.security.response.OAuth2ResponseFactory;
import org.capstone.maru.service.MemberAccountService;
import org.capstone.maru.service.RecommendService;
import org.springframework.security.oauth2.client.userinfo.DefaultOAuth2UserService;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserRequest;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserService;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

@Slf4j
@RequiredArgsConstructor
@Service
public class CustomOAuth2UserService implements OAuth2UserService<OAuth2UserRequest, OAuth2User> {

    private final MemberAccountService memberAccountService;

    private final RecommendService recommendService;

    @Override
    public OAuth2User loadUser(OAuth2UserRequest userRequest) throws OAuth2AuthenticationException {
        log.info("[Debug] OAuth2UserService loadUser!");

        OAuth2UserService<OAuth2UserRequest, OAuth2User> delegate = new DefaultOAuth2UserService();
        OAuth2User oAuth2User = delegate.loadUser(userRequest);

        String registrationId = userRequest.getClientRegistration().getRegistrationId();

        OAuth2Response extractAttributes = OAuth2ResponseFactory.getOAuth2Response(
            registrationId,
            oAuth2User.getAttributes()
        );

        return createMemberPrincipal(registrationId, extractAttributes,
            oAuth2User.getAttributes());
    }

    private String getMemberId(String registrationId, OAuth2Response oAuth2Response) {
        return registrationId + "_" + oAuth2Response.id();
    }

    private MemberPrincipal createMemberPrincipal(
        String registrationId,
        OAuth2Response extractAttributes,
        Map<String, Object> oauth2Attributes
    ) {
        String memberId = getMemberId(registrationId, extractAttributes);

        return MemberPrincipal.from(
            memberAccountService.login(
                memberId,
                extractAttributes.email(),
                extractAttributes.nickname(),
                extractAttributes.birthYear(),
                extractAttributes.gender().name(),
                extractAttributes.phoneNumber()
            ),
            oauth2Attributes
        );
    }
}
