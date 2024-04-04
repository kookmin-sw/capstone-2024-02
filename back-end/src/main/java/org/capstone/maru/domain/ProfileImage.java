package org.capstone.maru.domain;

import jakarta.persistence.Entity;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@ToString(callSuper = true)
@Entity
public class ProfileImage extends Image {

    // -- 생성자 메서드 -- //
    private ProfileImage(String fileName, String createdBy) {
        super(fileName);
        this.createdBy = createdBy;
        this.modifiedBy = createdBy;
    }

    public static ProfileImage of(
        String fileName,
        String createdBy
    ) {
        return new ProfileImage(fileName, createdBy);
    }

    /*
     * 기본 이미지를 생성합니다.
     * 이때 확장자를 포람하여 저장합니다.
     * 파일 이름만 저장하게되면 확장자를 알 수 없기 때문에,
     *
     * 유형 없이 저장 할 수 있지만, 권장인지는 모르겠습니다.
     */
    public static ProfileImage defaultImage(String memberId) {
        return new ProfileImage(memberId + "default.png", "system");
    }

    // -- Equals & Hash -- //
    @Override
    public boolean equals(Object o) {
        return super.equals(o);
    }

    @Override
    public int hashCode() {
        return super.hashCode();
    }
}
