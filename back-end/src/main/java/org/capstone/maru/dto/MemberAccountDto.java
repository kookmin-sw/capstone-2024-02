package org.capstone.maru.dto;

import java.time.LocalDateTime;
import lombok.Builder;
import org.capstone.maru.domain.MemberAccount;

@Builder
public record MemberAccountDto(
    String memberId,
    String email,
    String nickname,
    String birthYear,
    String gender,
    String phoneNumber,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy,
    Boolean initialized
) {

    public static MemberAccountDto from(MemberAccount entity) {
        return MemberAccountDto
            .builder()
            .memberId(entity.getMemberId())
            .email(entity.getEmail())
            .nickname(entity.getNickname())
            .birthYear(entity.getBirthYear())
            .gender(entity.getGender())
            .phoneNumber(entity.getPhoneNumber())
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .initialized(entity.getInitialized())

            .build();
    }

    public MemberAccount toEntity() {
        return MemberAccount.of(
            memberId,
            email,
            nickname,
            birthYear,
            gender,
            phoneNumber,
            createdBy
        );
    }
}