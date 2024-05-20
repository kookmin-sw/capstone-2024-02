package org.capstone.maru.domain.jsonb;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import jakarta.persistence.Embeddable;
import java.io.Serializable;
import java.util.Objects;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@JsonIgnoreProperties(ignoreUnknown = true)
@Embeddable
public class MemberFeatures implements Serializable {

    private String smoking;

    private String roomSharingOption;

    private Short mateAge;

    private String options;

    // -- 생성자 메서드 -- //

    private MemberFeatures(String smoking, String roomSharingOption, Short mateAge,
        String options) {
        this.smoking = smoking;
        this.roomSharingOption = roomSharingOption;
        this.mateAge = mateAge;
        this.options = options;
    }

    public static MemberFeatures of(
        String smoking,
        String roomSharingOption,
        Short mateAge,
        String options
    ) {
        return new MemberFeatures(
            smoking,
            roomSharingOption,
            mateAge,
            options
        );
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof MemberFeatures that)) {
            return false;
        }
        return Objects.equals(smoking, that.smoking) && Objects.equals(
            roomSharingOption, that.roomSharingOption) && Objects.equals(mateAge,
            that.mateAge) && Objects.equals(options, that.options);
    }

    @Override
    public int hashCode() {
        return Objects.hash(smoking, roomSharingOption, mateAge, options);
    }
}
