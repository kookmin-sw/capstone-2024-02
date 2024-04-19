import axios from 'axios';
import { NextResponse, type NextRequest } from 'next/server';

import { type FromAddrToCoordDTO } from '@/features/geocoding';

export async function GET(
  request: NextRequest,
  { params: { query } }: { params: { query: string } },
) {
  try {
    const response = await axios.get<FromAddrToCoordDTO>(
      `https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query=${query}`,
      {
        headers: {
          'X-NCP-APIGW-API-KEY-ID': process.env.NEXT_PUBLIC_NAVER_MAP_CLIENT_ID,
          'X-NCP-APIGW-API-KEY':
            process.env.NEXT_PUBLIC_NAVER_MAP_CLIENT_SECRET,
        },
      },
    );

    return new NextResponse(JSON.stringify(response.data), { status: 200 });
  } catch (error) {
    return new NextResponse(JSON.stringify({ success: false }), {
      status: 400,
    });
  }
}
