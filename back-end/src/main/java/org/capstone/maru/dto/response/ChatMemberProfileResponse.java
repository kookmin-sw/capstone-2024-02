package org.capstone.maru.dto.response;

import lombok.Builder;

@Builder
public record ChatMemberProfileResponse(
    String memberId,
    String nickname,
    String profileImageUrl
) {

    public static ChatMemberProfileResponse from(String memberId, String nickname,
        String profileImageUrl) {
        return ChatMemberProfileResponse
            .builder()
            .memberId(memberId)
            .nickname(nickname)
            .profileImageUrl(profileImageUrl)
            .build();
    }
}
