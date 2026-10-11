// Phone photos are 3-15 MB and 12-48 MP. Shrinking them in the browser means the upload is
// quick, stays under the API limit, and the preview fits in sessionStorage (~5 MB quota).
// The API still does the real quality check and leaf crop; this only caps the size.

const UPLOAD_MAX_SIDE = 2048; // enough detail for the API to crop the leaf from
const UPLOAD_SKIP_BYTES = 1.5 * 1024 * 1024; // small files go up untouched
const UPLOAD_JPEG_QUALITY = 0.92;
const PREVIEW_MAX_SIDE = 640;
const PREVIEW_JPEG_QUALITY = 0.8;

async function drawScaled(
  bitmap: ImageBitmap,
  maxSide: number,
  quality: number
): Promise<Blob | null> {
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export type PreparedUpload = {
  file: File; // what is sent to the API
  previewDataUrl: string; // small, safe to keep in sessionStorage
  resized: boolean;
};

export async function prepareUpload(original: File): Promise<PreparedUpload> {
  try {
    // "from-image" applies the EXIF rotation so portrait phone photos stay upright.
    const bitmap = await createImageBitmap(original, {
      imageOrientation: "from-image",
    });
    try {
      const needsResize =
        original.size > UPLOAD_SKIP_BYTES ||
        Math.max(bitmap.width, bitmap.height) > UPLOAD_MAX_SIDE;
      let file = original;
      if (needsResize) {
        const blob = await drawScaled(bitmap, UPLOAD_MAX_SIDE, UPLOAD_JPEG_QUALITY);
        if (blob && blob.size < original.size) {
          file = new File([blob], original.name.replace(/\.\w+$/, "") + ".jpg", {
            type: "image/jpeg",
          });
        }
      }
      const previewBlob = await drawScaled(
        bitmap,
        PREVIEW_MAX_SIDE,
        PREVIEW_JPEG_QUALITY
      );
      return {
        file,
        previewDataUrl: await blobToDataUrl(previewBlob ?? file),
        resized: file !== original,
      };
    } finally {
      bitmap.close();
    }
  } catch {
    // Unsupported format or decode failure: let the API judge the original file.
    return {
      file: original,
      previewDataUrl: await blobToDataUrl(original),
      resized: false,
    };
  }
}
