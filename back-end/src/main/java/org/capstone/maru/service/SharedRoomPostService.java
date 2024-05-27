package org.capstone.maru.service;

import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.dto.SharedRoomPostDto;
import org.capstone.maru.repository.postgre.SharedRoomPostRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@RequiredArgsConstructor
@Service
@Transactional
public class SharedRoomPostService {

    private final SharedRoomPostRepository sharedRoomPostRepository;

    @Transactional(readOnly = true)
    public List<SharedRoomPostDto> getMySharedRoomPosts(
        String memberId
    ) {
        return sharedRoomPostRepository
            .findAllByPublisherAccount_MemberId(memberId)
            .stream()
            .map(SharedRoomPostDto::from)
            .toList();
    }
}
