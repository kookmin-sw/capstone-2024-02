package org.capstone.maru.dto;

import java.time.LocalDateTime;
import lombok.Builder;
import org.capstone.maru.domain.MemberAccount;

@Builder
public record MemberAccountWithCardsDto(
    String memberId,
    String email,
    String nickname,
    String birthYear,
    String gender,
    String phoneNumber,
    FeatureCardDto myCard,
    FeatureCardDto mateCard,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static MemberAccountWithCardsDto from(MemberAccount entity) {
        return MemberAccountWithCardsDto
            .builder()
            .memberId(entity.getMemberId())
            .email(entity.getEmail())
            .nickname(entity.getNickname())
            .birthYear(entity.getBirthYear())
            .gender(entity.getGender())
            .phoneNumber(entity.getPhoneNumber())
            .myCard(FeatureCardDto.from(entity.getMyCard()))
            .mateCard(FeatureCardDto.from(entity.getMateCard()))
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .build();
    }

}