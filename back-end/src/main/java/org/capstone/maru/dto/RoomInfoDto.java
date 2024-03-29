package org.capstone.maru.dto;

import java.io.Serializable;
import lombok.Builder;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.RoomInfo;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;

@Builder
public record RoomInfoDto(
    Long id,
    Address address,
    RoomType roomType,
    Short size,
    Short numberOfRoom,
    RentalType rentalType,
    Long price,
    Long monthlyFee,
    Long managementFee,
    Long expectedPayment,
    Short recruitmentCapacity
) {

    public static RoomInfoDto from(RoomInfo entity) {
        return RoomInfoDto
            .builder()
            .id(entity.getId())
            .address(entity.getAddress())
            .roomType(entity.getRoomType())
            .size(entity.getSize())
            .numberOfRoom(entity.getNumberOfRoom())
            .rentalType(entity.getRentalType())
            .price(entity.getPrice())
            .monthlyFee(entity.getMonthlyFee())
            .managementFee(entity.getManagementFee())
            .expectedPayment(entity.getExpectedPayment())
            .recruitmentCapacity(entity.getRecruitmentCapacity())
            .build();
    }

    public RoomInfo toEntity() {
        return RoomInfo.of(
            address,
            roomType,
            size,
            numberOfRoom,
            rentalType,
            price,
            monthlyFee,
            managementFee,
            expectedPayment,
            recruitmentCapacity
        );
    }
}