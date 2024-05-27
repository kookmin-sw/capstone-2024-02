package org.capstone.maru.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity
public class Follow {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id", nullable = false)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(
        name = "follower",
        referencedColumnName = "memberId"
    )
    private MemberAccount follower;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(
        name = "following",
        referencedColumnName = "memberId"
    )
    private MemberAccount following;

    // -- 생성자 -- //
    public Follow(MemberAccount follower, MemberAccount following) {
        this.follower = follower;
        this.following = following;
    }

    // -- Method -- //

}
