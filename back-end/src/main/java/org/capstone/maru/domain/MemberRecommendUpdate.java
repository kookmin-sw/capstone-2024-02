package org.capstone.maru.domain;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import java.time.Duration;
import java.time.LocalDateTime;
import lombok.Getter;

@Getter
@Entity
public class MemberRecommendUpdate {

    @Id
    private String memberId;

    private LocalDateTime recommendUpdateAt;

    public MemberRecommendUpdate(String memberId, LocalDateTime recommendUpdateAt) {
        this.memberId = memberId;
        this.recommendUpdateAt = LocalDateTime.now();
    }

    public Boolean haveToUpdate(LocalDateTime now) {

        Duration duration = Duration.between(recommendUpdateAt, now);

        // 차이가 30분인지 확인
        return duration.equals(Duration.ofMinutes(30));
    }
}
