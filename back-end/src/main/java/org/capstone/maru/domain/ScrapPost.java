package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import java.util.Objects;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(indexes = {
    @Index(columnList = "scrapper_member_id"),
    @Index(columnList = "scrapped_post_id")
})
@Entity
public class ScrapPost {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(
        name = "scrapper_member_id",
        referencedColumnName = "memberId",
        nullable = false
    )
    private MemberAccount scrapper;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(
        name = "scrapped_post_id",
        referencedColumnName = "id",
        nullable = false
    )
    private SharedRoomPost scrapped;

    @Column
    private Boolean isScrapped;


    // -- 생성자 메서드 -- //
    private ScrapPost(MemberAccount scrapper, SharedRoomPost scrapped) {
        this.scrapper = scrapper;
        this.scrapped = scrapped;
        this.isScrapped = true;
    }

    public static ScrapPost of(MemberAccount scrapper, SharedRoomPost scrapped) {
        return new ScrapPost(scrapper, scrapped);
    }

    // -- 비지니스 로직 -- //
    public void toggleScrap() {
        if (Boolean.TRUE.equals(this.isScrapped)) {
            this.isScrapped = false;
            return;
        }
        this.isScrapped = true;
    }


    // -- Equals & Hash -- //
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ScrapPost scrapPost)) {
            return false;
        }
        return id != null && id.equals(scrapPost.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
