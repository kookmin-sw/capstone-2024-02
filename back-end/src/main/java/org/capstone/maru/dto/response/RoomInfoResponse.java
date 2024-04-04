package org.capstone.maru.dto.response;

import lombok.Builder;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;
import org.capstone.maru.dto.RoomInfoDto;

@Builder
public record RoomInfoResponse(
    Long id,
    Address address,
    String roomType,
    Short size,
    Short numberOfRoom,
    Short recruitmentCapacity,
    String rentalType,
    Long price,
    Long managementFee,
    Long expectedPayment,
    Long monthlyFee
) {

    public static RoomInfoResponse from(RoomInfoDto dto) {
        return RoomInfoResponse
            .builder()
            .id(dto.id())
            .address(dto.address())
            .roomType(dto.roomType().getDescription())
            .size(dto.size())
            .numberOfRoom(dto.numberOfRoom())
            .recruitmentCapacity(dto.recruitmentCapacity())
            .rentalType(dto.rentalType().getDescription())
            .price(dto.price())
            .managementFee(dto.managementFee())
            .expectedPayment(dto.expectedPayment())
            .monthlyFee(dto.monthlyFee())
            .build();
    }
}