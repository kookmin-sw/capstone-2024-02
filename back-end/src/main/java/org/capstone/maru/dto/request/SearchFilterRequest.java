package org.capstone.maru.dto.request;

import com.google.gson.Gson;
import java.util.List;
import org.capstone.maru.domain.constant.FloorType;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;

public record SearchFilterRequest(
    List<RoomType> roomTypes,
    List<RentalType> rentalTypes,
    RangeRequest expectedPaymentRange,
    Boolean hasLivingRoom,
    Short numberOfRoom,
    Short numberOfBathRoom,
    RangeRequest roomSizeRange,
    List<FloorType> floorTypes,
    Boolean canPark,
    Boolean hasAirConditioner,
    Boolean hasRefrigerator,
    Boolean hasWasher,
    Boolean hasTerrace
) {

    public static SearchFilterRequest fromJson(String json) {
        return new Gson().fromJson(json, SearchFilterRequest.class);
    }

}
