package org.capstone.maru.dto.response;

import lombok.Builder;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.security.principal.MemberPrincipal;

@Builder
public record AuthResponse(
    String memberId,
    String email,
    String name,
    String birthYear,
    String gender,
    String phoneNumber,
    Boolean initialized,
    Boolean univCertified,
    Long myCardId,
    Long mateCardId
) {

    public static AuthResponse from(MemberPrincipal memberPrincipal, Boolean initialized) {

        return AuthResponse
            .builder()
            .memberId(memberPrincipal.memberId())
            .email(memberPrincipal.email())
            .name(memberPrincipal.nickname())
            .birthYear(memberPrincipal.birthYear())
            .gender(memberPrincipal.gender())
            .phoneNumber(memberPrincipal.phoneNumber())
            .initialized(initialized)
            .build();
    }

    public static AuthResponse from(MemberAccount memberAccount) {
        return AuthResponse
            .builder()
            .memberId(memberAccount.getMemberId())
            .email(memberAccount.getEmail())
            .name(memberAccount.getNickname())
            .birthYear(memberAccount.getBirthYear())
            .gender(memberAccount.getGender())
            .phoneNumber(memberAccount.getPhoneNumber())
            .initialized(memberAccount.getInitialized())
            .univCertified(memberAccount.getUnivName() != null)
            .myCardId(memberAccount.getMyCard().getId())
            .mateCardId(memberAccount.getMateCard().getId())
            .build();
    }
}
