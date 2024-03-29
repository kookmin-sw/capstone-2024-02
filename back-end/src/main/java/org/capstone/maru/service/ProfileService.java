package org.capstone.maru.service;

import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.MemberCard;
import org.capstone.maru.dto.MemberCardDto;
import org.capstone.maru.dto.MemberProfileDto;
import org.capstone.maru.dto.response.AuthResponse;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;


@RequiredArgsConstructor
@Service
@Slf4j
public class ProfileService {

    private final MemberAccountService memberAccountService;

    @Transactional
    public MemberCardDto updateMyCard(String memberId, List<String> myFeatures) {
        log.info("updateMyCard - memberId: {}, myFeatures: {}", memberId, myFeatures);

        MemberAccount memberAccount = memberAccountService.searchMemberAccount(memberId);
        MemberCard myCard = memberAccount.getMyCard();

        memberAccount.updateInitialized(myFeatures);
        myCard.updateMemberFeatures(myFeatures);

        return MemberCardDto.from(myCard);
    }

    @Transactional(readOnly = true)
    public MemberProfileDto getMemberCard(String memberId, MemberPrincipal memberPrincipal) {
        log.info("getMyCard - memberId: {}", memberId);

        MemberCard memberCard = memberAccountService.searchMemberAccount(memberId).getMyCard();
        log.info("memberCard: {}", memberCard.getMemberFeatures());
        AuthResponse authResponse = AuthResponse.from(memberPrincipal,
            memberCard.getMemberFeatures() == null || memberCard.getMemberFeatures().isEmpty());

        return MemberProfileDto.from(memberCard, authResponse);
    }

    @Transactional
    public MemberCardDto updateRoomCard(String memberId, String roomCardId, List<String> strings) {
        log.info("updateRoomCard - memberId: {}, roomCardId: {}, myFeatures: {}", memberId,
            roomCardId, strings);

        /*
         * TODO
         * 해당 roomCardId의 글이 로그인 한 사용자가 쓴 일치하는지 확인
         * 해당 roomCardId의 글이 존재하는지 확인, 존재하는 경우 그 카드를 가져옵니다.
         * 가져온 카드의 정보를 업데이트하고 저장합니다.
         * */

        return MemberCardDto.builder().build();
    }

}
