package org.capstone.maru.dto;

import lombok.Builder;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.dto.response.AuthResponse;

@Builder
public record MemberProfileDto(
    AuthResponse authResponse,
    String profileImage,
    FeatureCardDto myCard,
    FeatureCardDto mateCard
) {

    public static MemberProfileDto from(String imgURL, FeatureCard myCard,
        FeatureCard mateCard,
        AuthResponse authResponse) {
        return MemberProfileDto
            .builder()
            .profileImage(imgURL)
            .myCard(FeatureCardDto.from(myCard))
            .mateCard(FeatureCardDto.from(mateCard))
            .authResponse(authResponse)
            .build();
    }
}
