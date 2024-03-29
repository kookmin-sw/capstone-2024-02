package org.capstone.maru.dto.response;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.List;
import lombok.Builder;
import org.capstone.maru.dto.MemberAccountWithCardsDto;


@Builder
public record MemberAccountWithCardsResponse(
    String memberId,
    String email,
    String nickname,
    String birthYear,
    String gender,
    String phoneNumber,
    List<String> myCardFeatures,
    List<String> mateCardFeatures,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static MemberAccountWithCardsResponse from(MemberAccountWithCardsDto dto) {
        return MemberAccountWithCardsResponse
            .builder()
            .memberId(dto.memberId())
            .email(dto.email())
            .nickname(dto.nickname())
            .birthYear(dto.birthYear())
            .gender(dto.gender())
            .phoneNumber(dto.phoneNumber())
            .myCardFeatures(dto.myCard().myFeatures())
            .mateCardFeatures(dto.mateCard().myFeatures())
            .createdAt(dto.createdAt())
            .createdBy(dto.createdBy())
            .modifiedAt(dto.modifiedAt())
            .modifiedBy(dto.modifiedBy())
            .build();
    }

}