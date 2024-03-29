package org.capstone.maru.dto;

import lombok.Builder;
import org.capstone.maru.domain.MemberCard;
import org.capstone.maru.dto.response.AuthResponse;

@Builder
public record MemberProfileDto(
    MemberCardDto memberCard,
    AuthResponse authResponse
) {

    public static MemberProfileDto from(MemberCard memberCard,
        AuthResponse authResponse) {
        return MemberProfileDto
            .builder()
            .memberCard(MemberCardDto.from(memberCard))
            .authResponse(authResponse)
            .build();
    }
}
