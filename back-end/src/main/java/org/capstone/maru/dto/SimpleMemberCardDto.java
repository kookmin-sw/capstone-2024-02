package org.capstone.maru.dto;

import java.util.List;
import lombok.Builder;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.ProfileImage;
import org.capstone.maru.domain.jsonb.MemberFeatures;

@Builder
public record SimpleMemberCardDto(
    String memberId,
    String nickname,
    String profileImageUrl,
    String location,
    MemberFeatures options,
    Integer score
) {

    public static SimpleMemberCardDto from(MemberAccount memberAccount, FeatureCard featureCard,
        ProfileImage profileImage, Integer score) {
        return SimpleMemberCardDto.builder()
            .memberId(memberAccount.getMemberId())
            .nickname(memberAccount.getNickname())
            .profileImageUrl(profileImage.getFileName())
            .location(featureCard.getLocation())
            .options(featureCard.getMemberFeatures())
            .score(score)
            .build();
    }
}
