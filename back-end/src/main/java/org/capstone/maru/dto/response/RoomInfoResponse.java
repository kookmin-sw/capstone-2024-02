package org.capstone.maru.dto.response;

import lombok.Builder;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;
import org.capstone.maru.domain.jsonb.ExtraOption;
import org.capstone.maru.dto.RoomInfoDto;

@Builder
public record RoomInfoResponse(
    Long id,
    Address address,
    String roomType,
    String floorType,
    Short size,
    Short numberOfRoom,
    Short numberOfBathRoom,
    Boolean hasLivingRoom,
    Short recruitmentCapacity,
    String rentalType,
    Long expectedPayment,
    ExtraOption extraOption
) {

    public static RoomInfoResponse from(RoomInfoDto dto) {
        return RoomInfoResponse
            .builder()
            .id(dto.id())
            .address(dto.address())
            .roomType(dto.roomType().getDescription())
            .floorType(dto.floorType().getDescription())
            .size(dto.size())
            .numberOfRoom(dto.numberOfRoom())
            .numberOfBathRoom(dto.numberOfBathRoom())
            .hasLivingRoom(dto.hasLivingRoom())
            .recruitmentCapacity(dto.recruitmentCapacity())
            .rentalType(dto.rentalType().getDescription())
            .expectedPayment(dto.expectedPayment())
            .extraOption(dto.extraOption())
            .build();
    }
}