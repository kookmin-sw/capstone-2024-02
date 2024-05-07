package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Convert;
import jakarta.persistence.Embedded;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import java.util.Objects;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.capstone.maru.domain.constant.FloorType;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;
import org.capstone.maru.domain.converter.FloorTypeConverter;
import org.capstone.maru.domain.converter.RentalTypeConverter;
import org.capstone.maru.domain.converter.RoomTypeConverter;
import org.capstone.maru.domain.jsonb.ExtraOption;
import org.capstone.maru.dto.RoomInfoDto;
import org.hibernate.annotations.DynamicUpdate;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

@Getter
@ToString(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity
@DynamicUpdate
public class RoomInfo {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Embedded
    private Address address;

    @Convert(converter = RoomTypeConverter.class)
    @Column
    private RoomType roomType;

    @Convert(converter = FloorTypeConverter.class)
    @Column
    private FloorType floorType;

    @Column
    private Short size;

    @Column
    private Short numberOfRoom;

    @Column
    private Short numberOfBathRoom;

    @Column
    private Boolean hasLivingRoom;

    @Convert(converter = RentalTypeConverter.class)
    @Column
    private RentalType rentalType;

    @Column(nullable = false)
    private Long expectedPayment;

    @Column
    private Short recruitmentCapacity;

    @JdbcTypeCode(SqlTypes.JSON)
    @Embedded
    private ExtraOption extraOption;

    // -- 생성 메서드 -- //
    private RoomInfo(Address address, RoomType roomType, FloorType floorType, Short size,
        Short numberOfRoom, Short numberOfBathRoom, Boolean hasLivingRoom, RentalType rentalType,
        Long expectedPayment, Short recruitmentCapacity, ExtraOption extraOption) {
        this.address = address;
        this.roomType = roomType;
        this.floorType = floorType;
        this.size = size;
        this.numberOfRoom = numberOfRoom;
        this.numberOfBathRoom = numberOfBathRoom;
        this.hasLivingRoom = hasLivingRoom;
        this.rentalType = rentalType;
        this.expectedPayment = expectedPayment;
        this.recruitmentCapacity = recruitmentCapacity;
        this.extraOption = extraOption;
    }

    public static RoomInfo of(
        Address address,
        RoomType roomType,
        FloorType floorType,
        Short size,
        Short numberOfRoom,
        Short numberOfBathRoom,
        Boolean hasLivingRoom,
        RentalType rentalType,
        Long expectedPayment,
        Short recruitmentCapacity,
        ExtraOption extraOption
    ) {
        return new RoomInfo(
            address,
            roomType,
            floorType,
            size,
            numberOfRoom,
            numberOfBathRoom,
            hasLivingRoom,
            rentalType,
            expectedPayment,
            recruitmentCapacity,
            extraOption
        );
    }

    // -- 비지니스 로직 -- //
    public void updateRoomInfo(RoomInfoDto dto) {
        updateAddress(dto.address());
        updateRoomType(dto.roomType());
        updateFloorType(dto.floorType());
        updateSize(dto.size());
        updateNumberOfRoom(dto.numberOfRoom());
        updateNumberOfBathRoom(dto.numberOfBathRoom());
        updateHasLivingRoom(dto.hasLivingRoom());
        updateRentalType(dto.rentalType());
        updateExpectedPayment(dto.expectedPayment());
        updateRecruitmentCapacity(dto.recruitmentCapacity());
        updateExtraOptions(dto.extraOption());
    }

    private void updateAddress(Address address) {
        if (Objects.equals(address, this.address)) {
            return;
        }
        this.address = address;
    }

    private void updateRoomType(RoomType roomType) {
        if (this.roomType.getCode() == roomType.getCode()) {
            return;
        }
        this.roomType = roomType;
    }

    private void updateFloorType(FloorType floorType) {
        if (this.floorType.getCode() == floorType.getCode()) {
            return;
        }
        this.floorType = floorType;
    }

    private void updateSize(Short size) {
        if (Objects.equals(this.size, size)) {
            return;
        }
        this.size = size;
    }

    private void updateNumberOfRoom(Short numberOfRoom) {
        if (Objects.equals(this.numberOfRoom, numberOfRoom)) {
            return;
        }
        this.numberOfRoom = numberOfRoom;
    }

    private void updateNumberOfBathRoom(Short numberOfBathRoom) {
        if (Objects.equals(this.numberOfBathRoom, numberOfBathRoom)) {
            return;
        }
        this.numberOfBathRoom = numberOfBathRoom;
    }

    private void updateHasLivingRoom(Boolean hasLivingRoom) {
        if (Objects.equals(this.hasLivingRoom, hasLivingRoom)) {
            return;
        }
        this.hasLivingRoom = hasLivingRoom;
    }

    private void updateRentalType(RentalType rentalType) {
        if (this.rentalType.getCode() == RoomInfo.this.rentalType.getCode()) {
            return;
        }
        this.rentalType = rentalType;
    }

    private void updateExpectedPayment(Long expectedPayment) {
        if (Objects.equals(this.expectedPayment, expectedPayment)) {
            return;
        }
        this.expectedPayment = expectedPayment;
    }

    private void updateRecruitmentCapacity(Short recruitmentCapacity) {
        if (Objects.equals(this.recruitmentCapacity, recruitmentCapacity)) {
            return;
        }
        this.recruitmentCapacity = recruitmentCapacity;
    }

    private void updateExtraOptions(ExtraOption extraOption) {
        if (Objects.equals(this.extraOption, extraOption)) {
            return;
        }
        this.extraOption = extraOption;
    }


    // -- Equals & Hash -- //
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof RoomInfo roomInfo)) {
            return false;
        }
        return id != null && id.equals(roomInfo.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
