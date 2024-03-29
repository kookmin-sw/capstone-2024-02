package org.capstone.maru.dto.response;

import java.time.LocalDateTime;
import lombok.Builder;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.dto.StudioRoomPostDto;

// TODO: 방 이미지 추가하기

/**
 * 게시판에서 보는 게시글 response
 */
@Builder
public record StudioRoomPostResponse(
    Long id,
    String title,
    String content,
    RoomImageResponse thumbnail,
    MemberAccountResponse publisherAccount,
    RoomInfoResponse roomInfo,
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
            .roomInfo(RoomInfoResponse.from(dto.roomInfo()))
            .createdAt(dto.createdAt())
            .createdBy(dto.createdBy())
            .modifiedAt(dto.modifiedAt())
            .modifiedBy(dto.modifiedBy())
            .build();
    }
}