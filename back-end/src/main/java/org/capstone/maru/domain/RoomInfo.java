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
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;
import org.capstone.maru.domain.converter.RentalTypeConverter;
import org.capstone.maru.domain.converter.RoomTypeConverter;

@Getter
@ToString(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity
public class RoomInfo {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Embedded
    private Address address;

    @Convert(converter = RoomTypeConverter.class)
    @Column
    private RoomType roomType;

    @Column
    private Short size;

    @Column
    private Short numberOfRoom;

    @Convert(converter = RentalTypeConverter.class)
    @Column
    private RentalType rentalType;

    @Column
    private Long price;

    @Column
    private Long monthlyFee;

    @Column
    private Long managementFee;

    @Column
    private Long expectedPayment;

    @Column
    private Short recruitmentCapacity;

    // -- 생성 메서드 -- //
    private RoomInfo(Address address, RoomType roomType, Short size, Short numberOfRoom,
        RentalType rentalType, Long price, Long monthlyFee, Long managementFee,
        Long expectedPayment,
        Short recruitmentCapacity) {
        this.address = address;
        this.roomType = roomType;
        this.size = size;
        this.numberOfRoom = numberOfRoom;
        this.rentalType = rentalType;
        this.price = price;
        this.monthlyFee = monthlyFee;
        this.managementFee = managementFee;
        this.expectedPayment = expectedPayment;
        this.recruitmentCapacity = recruitmentCapacity;
    }

    public static RoomInfo of(
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
        return new RoomInfo(
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
