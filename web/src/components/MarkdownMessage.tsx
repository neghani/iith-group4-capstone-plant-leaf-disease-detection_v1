"use client";

import ReactMarkdown from "react-markdown";

const proseClass =
  "text-sm text-forest [&_p]:mt-2 [&_p:first-child]:mt-0 [&_strong]:font-semibold [&_em]:italic [&_ul]:mt-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:mt-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:mt-1";

export function MarkdownMessage({
  content,
  className = "",
}: {
  content: string;
  className?: string;
}) {
  return (
    <div className={`${proseClass} ${className}`.trim()}>
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
}
