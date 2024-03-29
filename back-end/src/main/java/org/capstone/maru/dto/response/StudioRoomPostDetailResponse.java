package org.capstone.maru.dto.response;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.Builder;
import org.capstone.maru.dto.StudioRoomPostDetailDto;
import org.capstone.maru.dto.StudioRoomPostDto;

/**
 * 상세 페이지에서 보는 상세 게시글 response
 */
@Builder
public record StudioRoomPostDetailResponse(
    Long id,
    String title,
    String content,
    Set<RoomImageResponse> roomImages,
    MemberAccountWithCardsResponse publisherAccount,
    RoomInfoResponse roomInfo,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static StudioRoomPostDetailResponse from(StudioRoomPostDetailDto dto) {
        return StudioRoomPostDetailResponse
            .builder()
            .id(dto.id())
            .title(dto.title())
            .content(dto.content())
            .roomImages(
                dto.roomImages()
                   .stream()
                   .map(RoomImageResponse::from)
                   .collect(Collectors.toSet())
            )
            .publisherAccount(MemberAccountWithCardsResponse.from(dto.publisherAccount()))
            .roomInfo(RoomInfoResponse.from(dto.roomInfo()))
            .createdAt(dto.createdAt())
            .createdBy(dto.createdBy())
            .modifiedAt(dto.modifiedAt())
            .modifiedBy(dto.modifiedBy())
            .build();
    }
}