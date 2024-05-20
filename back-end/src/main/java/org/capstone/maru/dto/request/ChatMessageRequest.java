package org.capstone.maru.dto.request;

import lombok.Builder;

@Builder
public record ChatMessageRequest(
    Long roomId,
    String sender,
    String message,
    String nickname
) {

}