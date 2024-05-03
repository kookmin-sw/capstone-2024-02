package org.capstone.maru.repository.postgre.projection;

import java.util.List;
import org.capstone.maru.domain.MemberAccount;

public interface ParticipantsView {

    List<MemberAccount> getMemberAccounts();
}
