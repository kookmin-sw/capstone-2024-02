package org.capstone.maru.dto;

import lombok.Builder;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.dto.response.AuthResponse;

@Builder
public record MemberProfileDto(
    AuthResponse authResponse,
    String profileImage,
    MemberCardDto myCard,
    MemberCardDto mateCard
) {

    public static MemberProfileDto from(String imgURL, FeatureCard myCard,
        FeatureCard mateCard,
        AuthResponse authResponse) {
        return MemberProfileDto
            .builder()
            .profileImage(imgURL)
            .myCard(MemberCardDto.from(myCard))
            .mateCard(MemberCardDto.from(mateCard))
            .authResponse(authResponse)
            .build();
    }
}
