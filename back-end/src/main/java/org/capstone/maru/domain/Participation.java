package org.capstone.maru.domain;

import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
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
    @Index(columnList = "recruited_member_id"),
    @Index(columnList = "recruits_post_id")
})
@Entity
public class Participation {

    @Id
    @GeneratedValue
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(
        name = "recruited_member_id",
        referencedColumnName = "memberId",
        nullable = false
    )
    private MemberAccount recruitedMemberAccount;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(
        name = "recruits_post_id",
        referencedColumnName = "id",
        nullable = false
    )
    private SharedRoomPost recruitsSharedRoomPost;


    // -- 생성자 메서드 -- //
    private Participation(MemberAccount recruitedMemberAccount,
        SharedRoomPost recruitsSharedRoomPost) {
        this.recruitedMemberAccount = recruitedMemberAccount;
        this.recruitsSharedRoomPost = recruitsSharedRoomPost;
    }

    public static Participation of(
        MemberAccount recruitedMemberAccount,
        SharedRoomPost recruitsSharedRoomPost
    ) {
        Participation result = new Participation(
            recruitedMemberAccount, recruitsSharedRoomPost
        );

        recruitsSharedRoomPost.addSharedRoomPostRecruits(result);
        return result;
    }

    // -- Equals & Hash -- //
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Participation participation)) {
            return false;
        }
        return id != null && id.equals(participation.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
