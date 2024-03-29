package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.DiscriminatorColumn;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Inheritance;
import jakarta.persistence.InheritanceType;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;

import java.util.Objects;
import java.util.Optional;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@ToString(callSuper = true)
@Table(indexes = {
    @Index(columnList = "id", unique = true),
    @Index(columnList = "title"),
    @Index(columnList = "publisherGender"),
    @Index(columnList = "createdAt"),
    @Index(columnList = "createdBy")
})
@Inheritance(strategy = InheritanceType.SINGLE_TABLE)
@DiscriminatorColumn
@Entity
public abstract class SharedRoomPost extends AuditingFields {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Setter
    @Column(nullable = false, length = 255)
    private String title;

    @Setter
    @Column(columnDefinition = "TEXT")
    private String content;

    @Column
    private String publisherGender;

    @PrePersist
    public void fillInPublisherGender() {
        publisherGender =
            Optional.of(SecurityContextHolder.getContext())
                    .map(SecurityContext::getAuthentication)
                    .filter(Authentication::isAuthenticated)
                    .map(Authentication::getPrincipal)
                    .map(MemberPrincipal.class::cast)
                    .map(MemberPrincipal::gender)
                    .orElseThrow(
                        () -> new IllegalArgumentException("인증되지 않은 방식의 사용자가 게시글을 작성하려 하였습니다.")
                    );
    }


    // -- 생성자 메서드 -- //
    protected SharedRoomPost(String title, String content, String publisherGender) {
        this.title = title;
        this.content = content;
        this.publisherGender = publisherGender;
    }


    // -- Equals & Hash -- //
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof SharedRoomPost sharedRoomPost)) {
            return false;
        }
        return id != null && id.equals(sharedRoomPost.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
