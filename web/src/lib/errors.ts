const KNOWN: Record<string, string> = {
  "Unsupported file type. Upload JPEG, PNG, or WebP.": "errors.unsupportedType",
  "Empty file uploaded.": "errors.emptyFile",
  "Quality check failed": "errors.qualityFailed",
  "Upload failed": "errors.uploadFailed",
  "Something went wrong": "errors.generic",
};

export function translateError(
  message: string,
  t: (key: string) => string
): string {
  if (!message) return t("errors.generic");
  if (message.startsWith("Image exceeds maximum size")) return t("errors.tooLarge");
  const key = KNOWN[message];
  return key ? t(key) : message;
}
