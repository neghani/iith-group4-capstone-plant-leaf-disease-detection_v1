import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../src/i18n/LocaleProvider";
import { en, leafKeys, te } from "../src/i18n";
import { AppShell } from "../src/components/AppShell";

vi.mock("next/link", () => ({
  default: ({
    href,
    children,
    className,
  }: {
    href: string;
    children: React.ReactNode;
    className?: string;
  }) => (
    <a href={href} className={className}>
      {children}
    </a>
  ),
}));

vi.mock("next/navigation", () => ({
  usePathname: () => "/check",
}));

describe("language catalogs", () => {
  it("gives Telugu a string for every English label", () => {
    const english = leafKeys(en).sort();
    const telugu = leafKeys(te).sort();
    expect(telugu).toEqual(english);
    expect(te.nav.check).not.toBe(en.nav.check);
  });
});

describe("language toggle", () => {
  it("switches the check navigation label to Telugu", async () => {
    const user = userEvent.setup();
    render(
      <LocaleProvider>
        <AppShell breadcrumb="Check a crop">
          <h1>Let's take a look at your crop</h1>
        </AppShell>
      </LocaleProvider>
    );

    expect(screen.getByRole("link", { name: "Check a crop" })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "తెలుగు" }));
    expect(screen.getByRole("link", { name: "పంటను పరిశీలించండి" })).toBeTruthy();
    expect(document.documentElement.lang).toBe("te");

    await user.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByRole("link", { name: "Check a crop" })).toBeTruthy();
    expect(document.documentElement.lang).toBe("en");
  });
});
