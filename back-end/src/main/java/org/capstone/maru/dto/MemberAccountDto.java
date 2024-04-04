package org.capstone.maru.dto;

import java.time.LocalDateTime;
import java.util.Set;
import lombok.Builder;
import org.capstone.maru.domain.Follow;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.MemberCard;
import org.capstone.maru.domain.ProfileImage;

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

    public MemberAccount toEntity(MemberCard myCard, MemberCard mateCard, Set<Follow> followers,
        Set<Follow> followings, ProfileImage profileImage) {
        return MemberAccount.of(
            memberId,
            email,
            nickname,
            birthYear,
            gender,
            phoneNumber,
            createdBy,
            initialized,
            myCard,
            mateCard,
            followers,
            followings,
            profileImage
        );
    }
}