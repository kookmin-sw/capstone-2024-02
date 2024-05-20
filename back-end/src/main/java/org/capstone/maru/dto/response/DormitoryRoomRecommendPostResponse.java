package org.capstone.maru.dto.response;

import java.time.LocalDateTime;
import java.util.List;
import lombok.Builder;
import org.capstone.maru.domain.Address;
import org.capstone.maru.dto.DormitoryRoomRecommendPostDto;
import org.capstone.maru.dto.MemberAccountDto;
import org.capstone.maru.dto.RoomImageDto;

@Builder
public record DormitoryRoomRecommendPostResponse(
    Long id,
    String title,
    String content,
    RoomImageResponse thumbnail,
    MemberAccountResponse publisherAccount,
    Address address,
    Short recruitmentCapacity,
    Boolean isScrapped,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy,
    Integer score
) {

    public static DormitoryRoomRecommendPostResponse from(
        DormitoryRoomRecommendPostDto dto
    ) {
        return DormitoryRoomRecommendPostResponse
            .builder()
            .id(dto.id())
            .title(dto.title())
            .content(dto.content())
            .thumbnail(RoomImageResponse.from(dto.thumbnail()))
            .publisherAccount(MemberAccountResponse.from(dto.publisherAccount()))
            .address(dto.address())
            .recruitmentCapacity(dto.recruitmentCapacity())
            .isScrapped(dto.isScrapped())
            .createdAt(dto.createdAt())
            .createdBy(dto.createdBy())
            .modifiedAt(dto.modifiedAt())
            .modifiedBy(dto.modifiedBy())
            .score(dto.score())
            .build();
    }
}
