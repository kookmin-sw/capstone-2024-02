package org.capstone.maru.dto;

import java.time.LocalDateTime;
import java.util.Set;
import lombok.Builder;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.domain.Follow;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.ProfileImage;

@Builder
public record MemberAccountDto(
    String memberId,
    String email,
    String nickname,
    String birthYear,
    String gender,
    String phoneNumber,
    String profileImage,
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
            .profileImage(entity.getProfileImage().getFileName())
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .initialized(entity.getInitialized())
            .build();
    }

    public MemberAccount toEntity(FeatureCard myCard, FeatureCard mateCard, Set<Follow> followers,
        Set<Follow> followings, ProfileImage profileImage, Boolean recommendOn) {
        return MemberAccount.of(
            memberId,
            email,
            nickname,
            birthYear,
            gender,
            phoneNumber,
            createdBy,
            initialized,
            recommendOn,
            myCard,
            mateCard,
            followers,
            followings,
            profileImage
        );
    }
}