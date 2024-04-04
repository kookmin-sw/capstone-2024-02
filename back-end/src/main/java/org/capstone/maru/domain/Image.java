package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.DiscriminatorColumn;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Inheritance;
import jakarta.persistence.InheritanceType;
import jakarta.persistence.Table;
import java.util.Objects;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@ToString(callSuper = true)
@Table(indexes = {
    @Index(columnList = "fileName", unique = true)
})
@Inheritance(strategy = InheritanceType.TABLE_PER_CLASS)
@Entity
public abstract class Image extends AuditingFields {

    @Id
    @Column(name = "file_name", nullable = false)
    private String fileName;

    // -- 생성자 메서드 -- //
    protected Image(String fileName) {
        this.fileName = fileName;
    }

    // -- Equals & Hash -- //
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Image image)) {
            return false;
        }
        return fileName != null && fileName.equals(image.fileName);
    }

    @Override
    public int hashCode() {
        return Objects.hash(fileName);
    }
}
