package org.capstone.maru.dto.response;

import java.time.LocalDateTime;
import lombok.Builder;
import org.capstone.maru.domain.Address;
import org.capstone.maru.dto.DormitoryRoomPostDto;

@Builder
public record DormitoryRoomPostResponse(
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
    String modifiedBy
) {

    public static DormitoryRoomPostResponse from(DormitoryRoomPostDto dto) {
        return DormitoryRoomPostResponse
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
            .build();
    }
}
