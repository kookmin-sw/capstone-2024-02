package org.capstone.maru.dto.response;

import lombok.Builder;
import org.capstone.maru.dto.SimpleMemberProfileDto;

@Builder
public record ParticipantResponse(
    String memberId,
    String nickname,
    String profileImageFileName,
    String birthYear,
    Boolean isScrapped
) {

    public static ParticipantResponse from(SimpleMemberProfileDto dto) {
        return ParticipantResponse
            .builder()
            .memberId(dto.memberId())
            .nickname(dto.nickname())
            .profileImageFileName(dto.profileImageFileName())
            .birthYear(dto.birthYear())
            .isScrapped(dto.isScrapped())
            .build();
    }

}
