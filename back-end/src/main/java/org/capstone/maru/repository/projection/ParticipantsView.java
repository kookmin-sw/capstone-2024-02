package org.capstone.maru.repository.projection;

import java.util.List;
import org.capstone.maru.domain.MemberAccount;

public interface ParticipantsView {

    List<MemberAccount> getMemberAccounts();
}
