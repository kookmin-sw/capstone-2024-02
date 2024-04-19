/** @type {import('next').NextConfig} */
const nextConfig = {
  // reactStrictMode: false,
  reactStrictMode: true,
  compiler: {
    styledComponents: true,
  },
  async rewrites() {
    return [
      {
        source: '/maru-api/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL}/:path*`,
      },
    ];
  },
  trailingSlash: false /* 만약, 후행 슬래시가 필요하다면 true, default → false */,
};

export default nextConfig;
