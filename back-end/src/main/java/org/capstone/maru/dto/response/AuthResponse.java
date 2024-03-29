package org.capstone.maru.dto.response;

import lombok.Builder;
import org.capstone.maru.security.principal.MemberPrincipal;

@Builder
public record AuthResponse(
    String memberId,
    String email,
    String name,
    String birthYear,
    String gender,
    String phoneNumber,
    Boolean initialized
) {

    public static AuthResponse from(MemberPrincipal memberPrincipal, Boolean initialized) {

        return AuthResponse.builder()
            .memberId(memberPrincipal.memberId())
            .email(memberPrincipal.email())
            .name(memberPrincipal.nickname())
            .birthYear(memberPrincipal.birthYear())
            .gender(memberPrincipal.gender())
            .phoneNumber(memberPrincipal.phoneNumber())
            .initialized(initialized)
            .build();
    }
}
