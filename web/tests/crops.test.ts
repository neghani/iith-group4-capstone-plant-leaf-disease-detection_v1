import { describe, expect, it } from "vitest";
import { MASTER_CROPS } from "../src/lib/crops";

describe("PlantVillage crop list", () => {
  it("lists the 14 PlantVillage crops for the live MobileNet pin", () => {
    expect(MASTER_CROPS).toHaveLength(14);
    expect(MASTER_CROPS).toContain("Bell Pepper");
    expect(MASTER_CROPS).toContain("Corn");
    expect(MASTER_CROPS).toContain("Blueberry");
    expect(MASTER_CROPS).toContain("Soybean");
    expect(MASTER_CROPS).not.toContain("Rice");
    expect(MASTER_CROPS).not.toContain("Sugarcane");
    expect(MASTER_CROPS).not.toContain("Wheat");
  });

  it("has unique crop names", () => {
    expect(new Set(MASTER_CROPS).size).toBe(MASTER_CROPS.length);
    for (const crop of MASTER_CROPS) {
      expect(crop.length).toBeGreaterThan(0);
    }
  });
});
