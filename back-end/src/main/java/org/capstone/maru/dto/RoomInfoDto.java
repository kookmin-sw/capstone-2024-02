package org.capstone.maru.dto;

import java.io.Serializable;
import lombok.Builder;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.RoomInfo;
import org.capstone.maru.domain.constant.FloorType;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;
import org.capstone.maru.domain.jsonb.ExtraOption;

@Builder
public record RoomInfoDto(
    Long id,
    RoomType roomType,
    FloorType floorType,
    Short size,
    Short numberOfRoom,
    Short numberOfBathRoom,
    Boolean hasLivingRoom,
    RentalType rentalType,
    Long expectedPayment,
    ExtraOption extraOption
) {

    public static RoomInfoDto from(RoomInfo entity) {
        return RoomInfoDto
            .builder()
            .id(entity.getId())
            .roomType(entity.getRoomType())
            .floorType(entity.getFloorType())
            .size(entity.getSize())
            .numberOfRoom(entity.getNumberOfRoom())
            .numberOfBathRoom(entity.getNumberOfBathRoom())
            .hasLivingRoom(entity.getHasLivingRoom())
            .rentalType(entity.getRentalType())
            .expectedPayment(entity.getExpectedPayment())
            .extraOption(entity.getExtraOption())
            .build();
    }

    public RoomInfo toEntity() {
        return RoomInfo.of(
            roomType,
            floorType,
            size,
            numberOfRoom,
            numberOfBathRoom,
            hasLivingRoom,
            rentalType,
            expectedPayment,
            extraOption
        );
    }
}