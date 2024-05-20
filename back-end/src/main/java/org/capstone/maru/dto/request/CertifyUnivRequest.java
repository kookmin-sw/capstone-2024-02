package org.capstone.maru.dto.request;

public record CertifyUnivRequest(
    String email,
    String univName,
    Integer code
) {

}
