package org.capstone.maru.repository;

import java.util.List;
import java.util.Optional;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.repository.projection.InitialView;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MemberAccountRepository extends JpaRepository<MemberAccount, String> {

    Optional<MemberAccount> findByEmail(String email);

    InitialView getInitializedById(String memberId);
}
