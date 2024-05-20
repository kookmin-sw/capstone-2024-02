package org.capstone.maru.dto.response;

import java.time.LocalDateTime;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.Address;
import org.capstone.maru.dto.StudioRoomPostDto;

/**
 * 게시판에서 보는 게시글 response
 */
@Slf4j
@Builder
public record StudioRoomPostResponse(
    Long id,
    String title,
    String content,
    RoomImageResponse thumbnail,
    MemberAccountResponse publisherAccount,
    Address address,
    RoomInfoResponse roomInfo,
    Boolean isScrapped,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static StudioRoomPostResponse from(StudioRoomPostDto dto) {
        return StudioRoomPostResponse
            .builder()
            .id(dto.id())
            .title(dto.title())
            .content(dto.content())
            .thumbnail(RoomImageResponse.from(dto.thumbnail()))
            .publisherAccount(MemberAccountResponse.from(dto.publisherAccount()))
            .address(dto.address())
            .roomInfo(RoomInfoResponse.from(dto.roomInfo(), dto.recruitmentCapacity()))
            .isScrapped(dto.isScrapped())
            .createdAt(dto.createdAt())
            .createdBy(dto.createdBy())
            .modifiedAt(dto.modifiedAt())
            .modifiedBy(dto.modifiedBy())
            .build();
    }
}