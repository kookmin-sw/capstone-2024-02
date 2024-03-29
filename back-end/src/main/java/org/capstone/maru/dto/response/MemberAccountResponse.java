package org.capstone.maru.dto.response;

import java.io.Serializable;
import java.time.LocalDateTime;
import lombok.Builder;
import org.capstone.maru.dto.MemberAccountDto;

@Builder
public record MemberAccountResponse(
    String memberId,
    String email,
    String nickname,
    String birthYear,
    String gender,
    String phoneNumber,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static MemberAccountResponse from(MemberAccountDto dto) {
        return MemberAccountResponse
            .builder()
            .memberId(dto.memberId())
            .email(dto.email())
            .nickname(dto.nickname())
            .birthYear(dto.birthYear())
            .gender(dto.gender())
            .phoneNumber(dto.phoneNumber())
            .createdAt(dto.createdAt())
            .createdBy(dto.createdBy())
            .modifiedAt(dto.modifiedAt())
            .modifiedBy(dto.modifiedBy())
            .build();
    }
}