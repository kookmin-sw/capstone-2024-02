package org.capstone.maru.dto.response;

import java.util.List;
import lombok.Builder;
import org.capstone.maru.domain.jsonb.MemberFeatures;
import org.capstone.maru.dto.SimpleMemberCardDto;

@Builder
public record SimpleMemberCardResponse(
    String memberId,
    String nickname,
    String profileImageUrl,
    String location,
    MemberFeatures options,
    Integer score
) {

    public static SimpleMemberCardResponse from(SimpleMemberCardDto simpleMemberCardDto) {
        return SimpleMemberCardResponse.builder()
            .memberId(simpleMemberCardDto.memberId())
            .nickname(simpleMemberCardDto.nickname())
            .profileImageUrl(simpleMemberCardDto.profileImageUrl())
            .location(simpleMemberCardDto.location())
            .options(simpleMemberCardDto.options())
            .score(simpleMemberCardDto.score())
            .build();
    }

}
