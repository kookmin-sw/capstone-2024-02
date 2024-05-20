package org.capstone.maru.dto.response;

import java.time.LocalDateTime;
import java.util.Comparator;
import java.util.List;
import lombok.Builder;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.jsonb.MemberFeatures;
import org.capstone.maru.dto.DormitoryRoomPostDetailDto;

@Builder
public record DormitoryRoomPostDetailResponse(
    Long id,
    String title,
    String content,
    MemberFeatures roomMateFeatures,
    List<ParticipantResponse> participants,
    List<RoomImageResponse> roomImages,
    MemberAccountResponse publisherAccount,
    Address address,
    Short recruitmentCapacity,
    Boolean isScrapped,
    Long scrapCount,
    Long viewCount,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static DormitoryRoomPostDetailResponse from(DormitoryRoomPostDetailDto dto) {
        return DormitoryRoomPostDetailResponse
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
            .address(dto.address())
            .recruitmentCapacity(dto.recruitmentCapacity())
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
