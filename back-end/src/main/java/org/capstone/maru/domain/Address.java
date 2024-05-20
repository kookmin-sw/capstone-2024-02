package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;
import java.util.Objects;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Embeddable
@Getter
public class Address {

    @Column
    private String oldAddress; // 지번 주소

    @Column(nullable = false)
    private String roadAddress; // 도로명 주소


    // -- 생성 메서드 -- //
    private Address(String oldAddress, String roadAddress) {
        this.oldAddress = oldAddress;
        this.roadAddress = roadAddress;
    }

    public static Address of(
        String oldAddress,
        String roadAddress
    ) {
        return new Address(
            oldAddress,
            roadAddress
        );
    }

    // -- Equals & Hash -- //
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Address address)) {
            return false;
        }
        return Objects.equals(oldAddress, address.oldAddress) && Objects.equals(
            roadAddress, address.roadAddress);
    }

    @Override
    public int hashCode() {
        return Objects.hash(oldAddress, roadAddress);
    }
}
