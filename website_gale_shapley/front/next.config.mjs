/** @type {import('next').NextConfig} */
const nextConfig = {
    output: 'standalone',
    experimental: {
        serverActions: {
            allowedOrigins: ["0.0.0.0:50011", "localhost:50011", "http://csariel.xyz:50011"]
        },
    }
};

export default nextConfig;
