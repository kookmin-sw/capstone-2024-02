package org.capstone.maru.dto.request;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.util.List;
import org.capstone.maru.annotation.ImageFilesConstraints;
import org.capstone.maru.domain.Address;
import org.capstone.maru.dto.DormitoryRoomPostDto;
import org.capstone.maru.dto.FeatureCardDto;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.dto.request.StudioRoomPostRequest.ImageFileData;
import org.hibernate.validator.constraints.Length;

public record DormitoryRoomPostRequest(
    @ImageFilesConstraints
    List<@Valid ImageFileData> imageFilesData,
    @Valid
    PostData postData,
    @Valid
    LocationData locationData,
    MemberFeatureRequest roomMateCardData,
    @Valid
    StudioRoomPostRequest.ParticipationData participationData
) {

    // -- 생성자 -- //
    public DormitoryRoomPostDto toBaseDormitoryRoomPostDto(String publisherGender) {
        return DormitoryRoomPostDto
            .builder()
            .title(postData.title())
            .content(postData.content())
            .publisherGender(publisherGender)
            .address(Address.of(locationData.oldAddress(), locationData.roadAddress()))
            .recruitmentCapacity(participationData.recruitmentCapacity())
            .build();
    }

    public FeatureCardDto toMemberCardDto() {
        return FeatureCardDto
            .builder()
            .location(roomMateCardData.location())
            .myFeatures(roomMateCardData.features())
            .build();
    }

    public List<RoomImageDto> toRoomImagesDto() {
        return imageFilesData
            .stream()
            .map(imageFileData ->
                RoomImageDto
                    .builder()
                    .fileName(imageFileData.fileName())
                    .isThumbnail(imageFileData.isThumbNail())
                    .order(imageFileData.order())
                    .build()
            )
            .toList();
    }

    // -- Nested -- //

    public record PostData(
        @NotBlank(message = "제목을 작성해야 합니다.")
        @Length(max = 50, message = "제목은 50자를 넘어갈 수 없습니다.")
        String title,
        @NotBlank(message = "상세 정보를 작성해야 합니다.")
        String content
    ) {

    }

    public record LocationData(
        String oldAddress,
        String roadAddress
    ) {

    }

    public record ParticipationData(
        @Min(value = 0, message = "모집 인원은 음수 일 수 없습니다.")
        @Max(value = 10, message = "모집 인원에 이상치 값이 입력 되었습니다.")
        Short recruitmentCapacity,
        @NotNull(message = "참가자를 반드시 넣어주세요.")
        List<String> participationMemberIds
    ) {

    }
}
