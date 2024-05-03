package org.capstone.maru.repository.postgre;

import java.util.List;
import java.util.Optional;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.repository.postgre.projection.InitialView;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MemberAccountRepository extends JpaRepository<MemberAccount, String> {

    Optional<MemberAccount> findByEmail(String email);

    InitialView getInitializedById(String memberId);

    Optional<MemberAccount> findByMemberId(String memberId);

    List<MemberAccount> findByEmailContaining(String word);
}
