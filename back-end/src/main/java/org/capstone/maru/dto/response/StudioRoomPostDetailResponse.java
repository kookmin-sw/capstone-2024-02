package org.capstone.maru.dto.response;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.Builder;
import org.capstone.maru.dto.MemberAccountDto;
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
    List<String> roomMateFeatures,
    List<ParticipantResponse> participants,
    List<RoomImageResponse> roomImages,
    MemberAccountResponse publisherAccount,
    RoomInfoResponse roomInfo,
    Boolean isScrapped,
    Long scrapCount,
    Long viewCount,
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
            .roomMateFeatures(dto.roomMateCard().myFeatures())
            .participants(
                dto.participants()
                   .stream()
                   .map(ParticipantResponse::from)
                   .toList()
            )
            .roomImages(
                dto.roomImages()
                   .stream()
                   .map(RoomImageResponse::from)
                   .sorted(Comparator.comparing(RoomImageResponse::order))
                   .toList()
            )
            .publisherAccount(MemberAccountResponse.from(dto.publisherAccount()))
            .roomInfo(RoomInfoResponse.from(dto.roomInfo()))
            .isScrapped(dto.isScrapped())
            .scrapCount(dto.scrapCount())
            .viewCount(dto.viewCount())
            .createdAt(dto.createdAt())
            .createdBy(dto.createdBy())
            .modifiedAt(dto.modifiedAt())
            .modifiedBy(dto.modifiedBy())
            .build();
    }
}