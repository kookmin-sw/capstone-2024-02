package org.capstone.maru.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.Builder;
import org.capstone.maru.domain.StudioRoomPost;

@Builder
public record StudioRoomPostDetailDto(
    Long id,
    String title,
    String content,
    String publisherGender,
    Set<RoomImageDto> roomImages,
    MemberAccountWithCardsDto publisherAccount,
    RoomInfoDto roomInfo,
    Boolean isScrapped,
    Long scrapCount,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static StudioRoomPostDetailDto from(StudioRoomPost entity, Boolean isScrapped,
        Long scrapCount) {
        return StudioRoomPostDetailDto
            .builder()
            .id(entity.getId())
            .title(entity.getTitle())
            .content(entity.getContent())
            .roomImages(
                entity.getRoomImages()
                      .stream()
                      .map(RoomImageDto::from)
                      .collect(Collectors.toSet())
            )
            .publisherGender(entity.getPublisherGender())
            .publisherAccount(MemberAccountWithCardsDto.from(entity.getPublisherAccount()))
            .roomInfo(RoomInfoDto.from(entity.getRoomInfo()))
            .isScrapped(isScrapped)
            .scrapCount(scrapCount)
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .build();
    }
}