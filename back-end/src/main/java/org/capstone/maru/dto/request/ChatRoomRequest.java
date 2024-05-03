package org.capstone.maru.dto.request;

import java.util.List;
import lombok.Builder;

@Builder
public record ChatRoomRequest(
    String roomName,
    List<String> members
) {

}
