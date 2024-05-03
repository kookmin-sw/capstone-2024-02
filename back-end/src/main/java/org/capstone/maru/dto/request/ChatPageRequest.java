package org.capstone.maru.dto.request;

public record ChatPageRequest(
    Long roomId,
    int page,
    int size
) {

}
