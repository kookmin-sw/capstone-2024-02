package org.capstone.maru.repository;


import org.capstone.maru.domain.Participation;
import org.capstone.maru.repository.projection.ParticipantsView;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

public interface ParticipationRepository extends
    JpaRepository<Participation, Long> {

    @Query("select p.recruitedMemberAccount from Participation p where p.recruitsSharedRoomPost.id = :recruitsSharedRoomPostId")
    ParticipantsView findParticipationViewsByRecruitsSharedRoomPostId(
        Long recruitsSharedRoomPostId);
}
