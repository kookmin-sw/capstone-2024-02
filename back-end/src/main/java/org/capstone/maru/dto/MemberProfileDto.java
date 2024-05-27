package org.capstone.maru.dto;

import java.util.List;
import lombok.Builder;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.dto.response.AuthResponse;
import org.capstone.maru.dto.response.SharedRoomPostResponse;

@Builder
public record MemberProfileDto(
    AuthResponse authResponse,
    String profileImage,
    FeatureCardDto myCard,
    FeatureCardDto mateCard,
    List<SharedRoomPostResponse> posts
) {

    public static MemberProfileDto from(String imgURL, FeatureCard myCard,
        FeatureCard mateCard,
        AuthResponse authResponse,
        List<SharedRoomPostResponse> posts
    ) {
        return MemberProfileDto
            .builder()
            .profileImage(imgURL)
            .myCard(FeatureCardDto.from(myCard))
            .mateCard(FeatureCardDto.from(mateCard))
            .authResponse(authResponse)
            .posts(posts)
            .build();
    }
}
