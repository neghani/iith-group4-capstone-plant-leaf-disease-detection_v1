"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BookOpen, History, ScanLine, Sprout } from "lucide-react";
import type { ReactNode } from "react";
import { useT } from "@/i18n/LocaleProvider";
import { LanguageToggle } from "@/components/LanguageToggle";

const nav = [
  { href: "/check", labelKey: "nav.check", icon: ScanLine },
  { href: "/history", labelKey: "nav.history", icon: History },
  { href: "/guide", labelKey: "nav.guide", icon: BookOpen },
];

export function AppShell({
  children,
  breadcrumb,
}: {
  children: ReactNode;
  breadcrumb?: string;
}) {
  const pathname = usePathname();
  const t = useT();

  return (
    <div className="flex min-h-screen bg-cream">
      <aside className="flex w-[240px] shrink-0 flex-col border-r border-[var(--color-border)] bg-white px-4 py-6">
        <Link href="/check" className="mb-8 flex items-center gap-3 px-2">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-forest text-white">
            <Sprout className="h-5 w-5" strokeWidth={2.2} />
          </span>
          <span className="font-brand text-lg font-semibold text-forest">
            CropCare AI
          </span>
        </Link>

        <p className="mb-3 px-3 text-[10px] font-semibold tracking-widest text-[var(--color-text-muted)]">
          {t("nav.eyebrow")}
        </p>

        <nav className="flex flex-col gap-1">
          {nav.map(({ href, labelKey, icon: Icon }) => {
            const label = t(labelKey);
            const active =
              pathname === href || (href === "/check" && pathname === "/");
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors ${
                  active
                    ? "bg-sage text-forest"
                    : "text-[var(--color-text-muted)] hover:bg-sage/50 hover:text-forest"
                }`}
              >
                <Icon className="h-[18px] w-[18px]" strokeWidth={2} />
                {label}
              </Link>
            );
          })}
        </nav>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between gap-4 border-b border-[var(--color-border)] bg-white px-8 py-3">
          <p className="text-sm text-[var(--color-text-muted)]">{breadcrumb}</p>
          <LanguageToggle />
        </header>
        <main className="flex-1 px-8 py-8">{children}</main>
      </div>
    </div>
  );
}
