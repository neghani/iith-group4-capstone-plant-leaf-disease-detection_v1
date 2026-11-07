import type { Metadata } from "next";
import { LocaleProvider } from "@/i18n/LocaleProvider";
import "@fontsource-variable/inter";
import "@fontsource-variable/fraunces";
import "@fontsource-variable/noto-sans-telugu";
import "./globals.css";

// Fonts are self-hosted via @fontsource packages (imported above) instead of
// next/font/google, which fetches font data from fonts.googleapis.com during
// `next build` and fails inside Docker builds without network access.
// globals.css maps these variables to the fontsource family names; Tailwind's
// font-sans/font-serif utilities keep consuming the same variables.

export const metadata: Metadata = {
  title: "CropCare AI",
  description: "Plant disease detection and farmer assistance",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans">
        <LocaleProvider>{children}</LocaleProvider>
      </body>
    </html>
  );
}
