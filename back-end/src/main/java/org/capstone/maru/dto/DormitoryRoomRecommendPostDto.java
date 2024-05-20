package org.capstone.maru.dto;

import java.time.LocalDateTime;
import java.util.List;
import org.capstone.maru.domain.Address;
import org.capstone.maru.repository.postgre.projection.ScrapPostView;

public record DormitoryRoomRecommendPostDto(
    Long id,
    String title,
    String content,
    String publisherGender,
    List<RoomImageDto> roomImages,
    MemberAccountDto publisherAccount,
    Address address,
    Short recruitmentCapacity,
    Boolean isScrapped,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy,
    Integer score
) {

    public static DormitoryRoomRecommendPostDto from(DormitoryRoomRecommendPost entity,
        List<ScrapPostView> scrapPostViews) {
        return new DormitoryRoomRecommendPostDto(
            entity.getId(),
            entity.getTitle(),
            entity.getContent(),
            entity.getPublisherGender(),
            entity.getRoomImages().stream()
                  .map(RoomImageDto::from)
                  .toList(),
            MemberAccountDto.from(entity.getPublisherAccount()),
            entity.getAddress(),
            entity.getRecruitmentCapacity(),
            scrapPostViews
                .stream()
                .filter(
                    scrapPostView -> scrapPostView.getScrappedId()
                                                  .equals(entity.getId()))
                .map(ScrapPostView::getIsScrapped)
                .findFirst()
                .orElse(false),
            entity.getCreatedAt(),
            entity.getCreatedBy(),
            entity.getModifiedAt(),
            entity.getModifiedBy(),
            entity.getScore()
        );
    }

    public RoomImageDto thumbnail() {
        return roomImages
            .stream()
            .filter(RoomImageDto::isThumbnail)
            .findAny()
            .orElseThrow(IllegalArgumentException::new);
    }
}
