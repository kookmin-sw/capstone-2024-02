package org.capstone.maru.dto.response;

import lombok.Builder;

@Builder
public record SimpleMemberProfileResponse(
    String memberId,
    String nickname,
    String profileImageUrl
) {

    public static SimpleMemberProfileResponse from(String memberId, String nickname,
        String profileImageUrl) {
        return SimpleMemberProfileResponse.builder()
            .memberId(memberId)
            .nickname(nickname)
            .profileImageUrl(profileImageUrl)
            .build();
    }
}
