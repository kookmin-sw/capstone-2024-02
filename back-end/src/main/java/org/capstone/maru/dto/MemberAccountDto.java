package org.capstone.maru.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Set;
import lombok.Builder;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.domain.Follow;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.MemberRoom;
import org.capstone.maru.domain.ProfileImage;

@Builder
public record MemberAccountDto(
    String memberId,
    String email,
    String nickname,
    String birthYear,
    String gender,
    String phoneNumber,
    String profileImageFileName,
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
            .profileImageFileName(entity.getProfileImage().getFileName())
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .initialized(entity.getInitialized())
            .build();
    }

    public MemberAccount toEntity(FeatureCard myCard, FeatureCard mateCard, Set<Follow> followers,
        Set<Follow> followings, ProfileImage profileImage, List<MemberRoom> chatRooms) {
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
            profileImage,
            chatRooms
        );
    }
}