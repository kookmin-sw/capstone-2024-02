package org.capstone.maru.dto;

import lombok.Builder;
import org.capstone.maru.domain.MemberAccount;

@Builder
public record SimpleMemberProfileDto(
    String memberId,
    String nickname,
    String profileImageFileName,
    String birthYear,
    Boolean isScrapped
) {

    public static SimpleMemberProfileDto from(MemberAccount entity, Boolean isScrapped) {
        return SimpleMemberProfileDto
            .builder()
            .memberId(entity.getMemberId())
            .nickname(entity.getNickname())
            .profileImageFileName(entity.getProfileImage().getFileName())
            .birthYear(entity.getBirthYear())
            .isScrapped(isScrapped)
            .build();
    }
}
